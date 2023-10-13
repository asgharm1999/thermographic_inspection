from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor

model = AutoModelForSemanticSegmentation.from_pretrained("runs/checkpoint-xxxx")
model.push_to_hub("domainName/modelName", commit_message="message")

imageProcessor = AutoImageProcessor.from_pretrained("backboneModelName")
imageProcessor.push_to_hub("domainName/modelName", commit_message="message")