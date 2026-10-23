from pathlib import Path

import gradio as gr
import numpy as np
import timm
import torch
from PIL import Image
from torch import nn


IMAGE_SIZE = 224
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_LABELS = {
    "akiec": "Kératose actinique / carcinome in situ (AKIEC)",
    "bcc": "Carcinome basocellulaire (BCC)",
    "bkl": "Kératose bénigne (BKL)",
    "df": "Dermatofibrome (DF)",
    "mel": "Mélanome (MEL)",
    "nv": "Nævus mélanocytaire (NV)",
    "vasc": "Lésion vasculaire (VASC)",
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CHECKPOINT_PATH = Path(__file__).parent / "models" / "best_backbone_model.pth"


class SkinClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=0
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(CLASS_NAMES)),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        return self.classifier(features)


def load_model() -> nn.Module:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {CHECKPOINT_PATH}. "
            "Ajoutez best_backbone_model.pth dans le dossier models."
        )

    model = SkinClassifier()
    state_dict = torch.load(
        CHECKPOINT_PATH, map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


MODEL = load_model()


def preprocess(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    return (tensor - MEAN) / STD


def scores_for(logits: torch.Tensor) -> dict[str, float]:
    probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
    return {
        CLASS_LABELS[class_name]: float(probability)
        for class_name, probability in zip(CLASS_NAMES, probabilities)
    }


def image_from_normalized(tensor: torch.Tensor) -> Image.Image:
    pixels = tensor.detach().cpu() * STD + MEAN
    pixels = pixels[0].permute(1, 2, 0).numpy()
    pixels = np.clip(pixels * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def fgsm_perturbation(
    image_tensor: torch.Tensor, epsilon: float
) -> torch.Tensor:
    original = image_tensor.detach()
    perturbed_input = original.clone().requires_grad_(True)
    MODEL.zero_grad(set_to_none=True)

    with torch.no_grad():
        pseudo_label = MODEL(original).argmax(dim=1)

    loss = nn.functional.cross_entropy(MODEL(perturbed_input), pseudo_label)
    loss.backward()
    gradient_sign = perturbed_input.grad.sign()

    lower_bound = (torch.zeros_like(MEAN) - MEAN) / STD
    upper_bound = (torch.ones_like(MEAN) - MEAN) / STD
    perturbed = original + epsilon * gradient_sign
    return torch.maximum(torch.minimum(perturbed, upper_bound), lower_bound)


def predict(
    image: Image.Image | None, simulate_fgsm: bool, epsilon: float
) -> tuple[dict[str, float], dict[str, float], Image.Image | None, str]:
    if image is None:
        raise gr.Error("Veuillez importer une image dermoscopique.")

    image_tensor = preprocess(image)
    with torch.inference_mode():
        clean_logits = MODEL(image_tensor)
    clean_scores = scores_for(clean_logits)

    if not simulate_fgsm:
        return (
            clean_scores,
            {},
            None,
            "Classification effectuée. La simulation FGSM est désactivée.",
        )

    adversarial_tensor = fgsm_perturbation(image_tensor, epsilon)
    with torch.inference_mode():
        adversarial_logits = MODEL(adversarial_tensor)

    return (
        clean_scores,
        scores_for(adversarial_logits),
        image_from_normalized(adversarial_tensor),
        (
            f"Simulation FGSM terminée (epsilon = {epsilon:.3f} dans l’espace "
            "normalisé). Il s’agit d’une perturbation d’étude, pas d’un "
            "détecteur d’attaque entraîné."
        ),
    )


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="slate"),
    title="Dermato-IA — démonstrateur de recherche",
) as demo:
    gr.Markdown(
        """
        # Dermato-IA
        ### Classification expérimentale de lésions cutanées

        Importez une image dermoscopique pour obtenir les scores des sept
        classes du modèle EfficientNet-B0. La simulation FGSM compare les
        scores avant et après une perturbation calculée à partir du modèle.
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil", image_mode="RGB", label="Image dermoscopique"
            )
            simulate_fgsm = gr.Checkbox(
                label="Afficher une simulation de perturbation FGSM",
                value=False,
            )
            epsilon = gr.Slider(
                minimum=0.0,
                maximum=0.1,
                value=0.03,
                step=0.005,
                label="Amplitude epsilon (espace normalisé)",
            )
            submit = gr.Button("Analyser l’image", variant="primary")
        with gr.Column():
            clean_output = gr.Label(
                num_top_classes=3, label="Scores — image originale"
            )
            adversarial_output = gr.Label(
                num_top_classes=3, label="Scores — après perturbation FGSM"
            )
            adversarial_image = gr.Image(
                label="Image perturbée (aperçu)", interactive=False
            )

    status = gr.Markdown()
    submit.click(
        fn=predict,
        inputs=[input_image, simulate_fgsm, epsilon],
        outputs=[clean_output, adversarial_output, adversarial_image, status],
    )

    gr.Markdown(
        """
        ---
        **Avertissement médical :** ce démonstrateur est un prototype de
        recherche fondé sur HAM10000. Il ne constitue ni un dispositif médical,
        ni un diagnostic, ni un avis clinique. Ses résultats ne doivent pas
        guider une décision de soin. Les scores ne sont pas des probabilités
        cliniquement calibrées. La sortie FGSM illustre la sensibilité du
        classifieur : le modèle chargé ne possède pas de tête de détection
        d’attaques entraînée.

        N’importez aucune image identifiante ni donnée médicale personnelle.
        """
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
