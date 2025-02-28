import os 
import numpy as np
import face_recognition


# images_folder = 'images'
# encoding_path='only_faces'


def generate_embeddings(images_folder, encoding_path='face_embeddings'):
    '''
    This function generates embeddings for the images in the images folder.
    The embeddings are saved in the only_faces folder.
    Args :
        images_folder : str
            The path to the folder containing the images.
        encoding_path : str (default : 'face_embeddings') - Optional 
            The path to the folder where the embeddings will be saved.
    '''
    assert isinstance(images_folder,str), 'The images folder path should be a string'
    
    if not os.path.exists(encoding_path):
        os.makedirs(encoding_path)
    
    if not os.listdir(images_folder):
        raise FileNotFoundError('The images folder is empty. Please provide images in the folder.')
    
    print(f'Reading images',end=' ')
    all_img_names=[i for i in os.listdir(images_folder) if i.endswith('.jpg') or i.endswith('.png') or i.endswith('.jpeg')]
    print(f'Found {len(all_img_names)} images')
    for i in os.listdir(images_folder):
        if i not in all_img_names:
            continue
        face_img=face_recognition.load_image_file(os.path.join(images_folder,i))
        embedding=face_recognition.face_encodings(face_img)[0]
        np.save(os.path.join(encoding_path,'embedding-'+i.split('.')[0]), embedding)
    
    print(f'Embeddings obtained and saved')
