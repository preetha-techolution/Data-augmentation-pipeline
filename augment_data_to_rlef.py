import os
import cv2
import json
import pandas as pd
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
import requests
import random

def json_creater(inputs, closed):
    data = []
    count = 1
    highContrastingColors = ['rgba(0,255,81,1)', 'rgba(255,219,0,1)', 'rgba(255,0,0,1)', 'rgba(0,4,255,1)', 
                             'rgba(227,0,255,1)']
    
    for index, input in enumerate(inputs):
        try:
            # Ensure the input is evaluated properly if it's a string representing a list of points
            points = eval(input)  # Safely convert string to list of tuples (e.g., '[[x1, y1], [x2, y2], ...]')
        except Exception as e:
            raise ValueError(f"Failed to evaluate input '{input}': {e}")
        
        color = random.sample(highContrastingColors, 1)[0]
        json_id = count
        sub_json_data = {}
        sub_json_data['id'] = json_id
        sub_json_data['name'] = json_id
        sub_json_data['color'] = color
        sub_json_data['isClosed'] = closed
        sub_json_data['selectedOptions'] = [{"id": "0", "value": "root"},
                                            {"id": str(random.randint(10, 20)), "value": inputs[input]}]
        
        # Create vertices for each point in the evaluated list
        vertices = []
        is_first = True
        for vertex in points:
            vertex_json = {}
            if is_first:
                vertex_json['id'] = json_id
                vertex_json['name'] = json_id
                is_first = False
            else:
                json_id = count
                vertex_json['id'] = json_id
                vertex_json['name'] = json_id
            vertex_json['x'] = vertex[0]
            vertex_json['y'] = vertex[1]
            vertices.append(vertex_json)
            count += 1
        
        sub_json_data['vertices'] = vertices
        data.append(sub_json_data)
    
    return json.dumps(data)



def multipointPath_To_WeirdAutoaiAnnotationFormat(annotations):
    li = {}
    
    for ann in annotations:
        # Extract the coordinates from the 'vertices' list
        vertices = ann['vertices']
        label = ann['label']
        
        # Check that there are exactly 4 vertices
        if len(vertices) != 4:
            raise ValueError(f"Expected 4 vertices, but found {len(vertices)} for annotation: {ann}")
        
        # Format the annotation as a string using x and y from each vertex
        formatted_coords = f"[[{vertices[0]['x']}, {vertices[0]['y']}], " \
                           f"[{vertices[1]['x']}, {vertices[1]['y']}], " \
                           f"[{vertices[2]['x']}, {vertices[2]['y']}], " \
                           f"[{vertices[3]['x']}, {vertices[3]['y']}]]"
        
        li[formatted_coords] = label
    
    # Pass the dictionary to the json_creater function
    rlef_format = json_creater(li, True)
    
    print("rlef fomat", rlef_format)  # For debugging
    return rlef_format





def send_to_rlef(img_path, model_id, tag, label, annotation, confidence_score=100, prediction='predicted'):
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource"
    payload = {
        'model': model_id,
        'status': 'backlog',
        'csv': 'csv',
        'label': label,
        'tag': tag,
        'model_type': 'imageAnnotation',
        'prediction': prediction,
        'confidence_score': confidence_score,
        'imageAnnotations': str(annotation)
    }
    files = [('resource', (f'{img_path}', open((os.path.join('augmented_images', img_path)), 'rb'), 'image/png'))]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    print(f"Response for {img_path}: {response.status_code}")
    if response.status_code == 200:
        print(f'Successfully uploaded to AutoAI')
    else:
        print(f'Error while uploading to  AutoAI')
        print(response.text)

def extract_label_from_annotation(annotation):
    """Extract the label from the 'selectedOptions' attribute of the annotation."""
    selected_options = annotation.get('selectedOptions', [])
    
    # Default label
    label = 'Label'
    
    # Check if there are at least two selected options and extract the label
    if len(selected_options) > 1:
        label = selected_options[1]['value']  # Access the second value attribute
        print("===========================================",label)
    
    return label


def process_images(input_csv_path, images_folder_path, num_augmentations=5):
    if not os.path.exists('augmented_images'):
        os.makedirs('augmented_images')
    
    df = pd.read_csv(input_csv_path)
    augmenters = {
        #"Multiply": iaa.Multiply((1.2, 1.5)),
        #"Affine": iaa.Affine(rotate=10, scale=(0.5, 0.7)),
        "Rotate" : iaa.Rotate((-90, 90)),
        "Noise": iaa.Add((-40, 40)),
        "Brightness": iaa.WithBrightnessChannels(
    iaa.Add((-50, 50)), to_colorspace=[iaa.CSPACE_Lab, iaa.CSPACE_HSV]),
        "Hue" : iaa.MultiplyHue((0.5, 1.5)),
        "Saturation" : iaa.MultiplySaturation((0.5, 1.5)),
         "Contrast" : iaa.GammaContrast((0.5, 2.0)),
        "Fliplr" : iaa.Fliplr(0.5),
        "Flipud" : iaa.Flipud(0.5),
        "Shear" : iaa.Affine(shear=(-20, 20)),
        "Translate" : iaa.TranslateX(px=(-90, 90)),
        "Sharpness" : iaa.pillike.EnhanceSharpness()
        }

    ia.seed(1)
    model_id = "6731e6f37b9b1be3bdf4e904"
    
    for idx, row in df.iterrows():
        image_name = row['name']
        label = row['label']
        tag = row['tag']
        image_path = os.path.join(images_folder_path, image_name)

        if not os.path.isfile(image_path):
            print(f"Image {image_name} not found. Skipping.")
            continue

        annotations = json.loads(row['imageAnnotations'].replace('\"', '"'))
        #print("annotations=====", annotations)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image {image_name}. Skipping.")
            continue

        all_keypoints = []
        for annotation in annotations:
            vertices = annotation['vertices']
            for vertex in vertices:
                all_keypoints.append(Keypoint(x=int(vertex['x']), y=int(vertex['y'])))

        kps = KeypointsOnImage(all_keypoints, shape=image.shape)

        augmented_rows = []
        for augmenter_name, augmenter in augmenters.items():
            for i in range(num_augmentations):
                image_aug, kps_aug = augmenter(image=image, keypoints=kps)
                keypoint_idx = 0
                modified_annotations = []
                
                for annotation in annotations:
                    # Extract the label from the annotation
                    class_label = extract_label_from_annotation(annotation)
                    
                    vertices = annotation['vertices']
                    modified_vertices = []
                    for j in range(4):  # Assuming 4 keypoints per annotation
                        augmented_kp = kps_aug.keypoints[keypoint_idx]
                        modified_vertex = vertices[j].copy()
                        modified_vertex['x'] = int(augmented_kp.x)
                        modified_vertex['y'] = int(augmented_kp.y)
                        modified_vertices.append(modified_vertex)
                        keypoint_idx += 1
                    
                    modified_annotation = annotation.copy()
                    modified_annotation['vertices'] = modified_vertices
                    modified_annotation['label'] = class_label  # Add the label to the annotation
                    modified_annotations.append(modified_annotation)
                    print("modified=======", modified_annotations)

                output_img_name = f"{os.path.splitext(image_name)[0]}_{augmenter_name}_{i+1}.jpg"
                cv2.imwrite(os.path.join('augmented_images', output_img_name), image_aug)
                
                # Generate proper label array for each segment
                # Pass the modified annotations to the multipointPath_To_WeirdAutoaiAnnotationFormat function
                rlef_annotation = multipointPath_To_WeirdAutoaiAnnotationFormat(modified_annotations)

                send_to_rlef(output_img_name, model_id, tag, label, rlef_annotation)



def main():
    input_csv_path = 'dataSetCollection_dummy_resources.csv'
    images_folder_path = 'dummy_test'
    num_augmentations = 5
    process_images(input_csv_path, images_folder_path, num_augmentations)

if __name__ == "__main__":
    main()
