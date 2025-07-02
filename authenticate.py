import face_recognition
import os
from generate_face_embeddings import generate_embeddings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import onnx
import onnxruntime


parser = argparse.ArgumentParser(description='Pass user arguments for authentication')

# When this argument is passed along with the path containing the user images the face embeddings are generated.
parser.add_argument('--generate', type=str, default=None, help='Command to generate embeddings')
# parser.add_argument('file_path', type=str, default=None, help='Please provide the path to the images directory')


class AuthenticateUser():
    def __init__(self, face_encdoings_folder,models_folder,video_input=0):
        '''
        This class is used to authenticate the user using face recognition and anti-spoofing techniques.
        Args :
            face_encdoings_folder : str
                The path to the folder containing the face embeddings.
            models_folder : str
                The path to the folder containing the models.
            video_input : int (default : 0) - Optional
                The video input source. Default is the webcam.
        '''
        assert face_encdoings_folder, 'The path to the face embeddings folder is not provided. Please provide the path.'
        assert models_folder, 'The path to the models folder is not provided. Please provide the path.'
        assert os.path.exists(face_encdoings_folder), 'The face embeddings folder does not exist. Please provide the correct path.'
        assert os.path.exists(models_folder), 'The models folder does not exist. Please provide the correct path.'

        if not os.listdir(face_encdoings_folder):
            raise FileNotFoundError('There are no embeddings in the folder')
        
        if not os.listdir(models_folder):
            raise FileNotFoundError('There are no model weights inside the folder')

        # Load Face encodings
        self.known_faces = []
        self.known_names = []
        self.face_encdoings_folder = face_encdoings_folder
        self.models_folder = models_folder

        start= time.time()
        for i in os.listdir(face_encdoings_folder):
            self.known_names.append(i.split('-')[1].split('.')[0])
            self.known_faces.append(np.load(os.path.join(face_encdoings_folder,i)))
        
        print('Finished loading face embeddings and it took {} seconds'.format(time.time()-start))

        # Load models
        self.depth_model_name = 'anti_spoofing_quantized.onnx'

        start = time.time()
        onnx.checker.check_model(onnx.load(os.path.join(self.models_folder,self.depth_model_name)))
        self.depth_map_model = self.create_inference_session(os.path.join(self.models_folder,self.depth_model_name))
        print('Finished loading models and it took {} seconds'.format(time.time()-start))

        # Initialize the camera
        self.video_input = video_input
        self.video_capture = cv2.VideoCapture(video_input)

        self.is_authenticated = False
    
    def get_video_capture(self):
        '''
        This function returns the video capture object.
        '''
        return self.video_capture
    
    def create_inference_session(self,quantized_model_path,provider='CPUExecutionProvider'):
        """Create ONNX Runtime inference session"""
        # Set up ONNX Runtime options for Processor (Raspberry Pi)
        options = onnxruntime.SessionOptions()

        # options.intra_op_num_threads = 4  # Adjust based on your Raspberry Pi's CPU
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Create inference session
        session = onnxruntime.InferenceSession(
            quantized_model_path,
            options,
            providers=[provider] # ['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        return session
    
    def preprocess_image(self,image, target_size=(252, 252)):

        image = cv2.resize(image, target_size)
        if len(image.shape) == 2 or image.shape[2] == 1:
        # Replicate the single channel image across three channels
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        
        image = image.astype(np.float32) / 255.0
        
        image = np.transpose(image, (2, 0, 1))  
        image = np.expand_dims(image, axis=0)   
        
        return image

    def get_spoof_prediction(self, frame):
        '''
        This function takes a frame as input and returns the prediction of the anti-spoofing model.
        Args:
            frame : numpy array
                The frame to be processed.
        Returns:
            prediction : int
                The prediction of the anti-spoofing model.
        '''
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess the frame as a tensor for the classifier
        preprocessed_frame = self.preprocess_image(frame_rgb)

        depth_input = {self.depth_map_model.get_inputs()[0].name: preprocessed_frame}
        depth_map,output = self.depth_map_model.run(None, depth_input)
        
        prediction = np.argmax(output, axis=1)[0]
        
        return depth_map,prediction,frame_rgb
    
    def get_user_name(self,frame_rgb):
        '''
        This function takes a frame as input and returns the name of the user if authenticated.
        Args:
            frame_rgb : numpy array
                The frame to be processed.
        Returns:
            name : str
                The name of the user if authenticated, else None.
        '''
        face_locations = face_recognition.face_locations(frame_rgb)
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
        
        for face_encoding in face_encodings:
            results = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=0.5)
            for i in range(len(results)):
                if results[i]:
                    return self.known_names[i]
        return None
    
    def authenticate(self):
        '''
        This function authenticates the user using face recognition and anti-spoofing techniques.
        '''
        while True:
            start_time = time.time()
            ret, frame = self.video_capture.read()
            if ret:
                
                depth_map, prediction, frame_rgb = self.get_spoof_prediction(frame)
                
                # Display the resulting frame
                # cv2.imshow('Live Camera Feed', frame_rgb)
                # Display the depth map
                # cv2.imshow('Depth Map', depth_map)

                if prediction == 1:
                    # If the prediction is real, proceed with face recognition
                    print('Live User, Proceeding for face recognition')

                    # Find all the faces in the frame
                    auth = self.get_user_name(frame_rgb)
                    if auth is not None:
                        print(f'✅ User {auth} authenticated successfully')
                        self.is_authenticated = True
                    else:
                        print('❌ User not recognized, please try again')
                        self.is_authenticated = False 
                else:
                    print('Spoof detected')
                    self.is_authenticated = False
                end_time = time.time()
                print(f'Time taken for inference is {end_time-start_time}')
                start_time = end_time
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parser.parse_args()

    if args.generate is not None:
        if os.path.exists(args.generate):
            generate_embeddings(args.generate)
        else:
            raise FileNotFoundError('The path to the images directory is incorrect or not provided. Please provide the correct path.')
    else:
        if os.path.exists('face_embeddings'):
            auth = AuthenticateUser(os.path.join(os.getcwd(),'face_embeddings'),os.path.join(os.getcwd(),'models'))
            auth.authenticate()    
        else:
            raise FileNotFoundError('The face embeddings are not found. Please generate the embeddings first.')