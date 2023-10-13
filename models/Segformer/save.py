from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor

model = AutoModelForSemanticSegmentation.from_pretrained(
    "models/Segformer/runs/testing/checkpoint-159"  # TODO: Replace with checkpoint path
)
model.push_to_hub(
    "ChristopherS27/testModel", commit_message="testing"
)  # TODO: Replace with model name

imageProcessor = AutoImageProcessor.from_pretrained(
    "nvidia/mit-b0"
)  # This is image processor of backbone
imageProcessor.push_to_hub(
    "ChristopherS27/testModel", commit_message="message"
)  # TODO: Replace with model name
