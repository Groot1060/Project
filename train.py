import face_recognition
import os
import pickle

def train_model(data_dir=r'C:\Users\kisho\Downloads\p3\dataset'):

    known_face_encodings = []
    known_face_names = []

    # Iterate over each person in the dataset directory
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if os.path.isdir(person_dir):
            # Iterate over each image file for that person
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                if image_name.endswith(('.jpg', '.jpeg', '.png')):
                    # Load the image file and encode the face
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    
                    if encodings:
                        # Assuming one face per image, add the encoding
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(person_name)
                    else:
                        print(f"No faces found in {image_name}. Skipping.")

    # Save the known faces and their names
    with open('known_faces.pkl', 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)
        print(f"Model trained with {len(known_face_encodings)} faces.")

if __name__ == '__main__':
    train_model()