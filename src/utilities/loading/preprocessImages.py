import os
from PIL import Image
import torch
import csv
from transformers import YolosForObjectDetection, YolosImageProcessor


model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
image_processor = YolosImageProcessor.from_pretrained('hustvl/yolos-tiny')

idx = 0
image_path = "test_data/val2017"
if os.path.isdir( image_path ):
    files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
with open("test_data/cocoval2017.csv", "w+", newline="") as csvFile:
    writer = csv.writer(csvFile)
    for f in files:
        if (idx == 100):
            break
        image = Image.open(f"{image_path}/{f}", "r")
        pixel_values = list(image.getdata())
        if (len(pixel_values) != 640 * 426):
            continue

        inputs = image_processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits

        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
        bestRes = {'score': 0, 'label': -1}
        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            if (score > bestRes['score']):
                bestRes['score'] = score
                bestRes['label'] = label
        
        if (bestRes['label'] == -1): continue
        
        p_vals = [bestRes['label'].item()]
        for tup in pixel_values:
            p_vals.append(tup[0])
            p_vals.append(tup[1])
            p_vals.append(tup[2])
        writer.writerow(p_vals)
        idx += 1
print(f"Wrote {idx} images to csv.")