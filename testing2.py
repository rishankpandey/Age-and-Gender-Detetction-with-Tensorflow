from keras.models import model_from_json
import cv2
import argparse
import numpy as np
import keras

class FaceCV(object):
    """
    Singleton class for face recongnition task
    """
    CASE_PATH = "haarcascade_frontalface_default.xml"
    


    def __new__(cls, weight_file=None,face_size=48):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self,face_size=48):
        self.face_size = face_size
        #file = open("json_file", 'r',errors='ignore')
        #model_json = file.read()
        #file.close()
        #self.loaded_model = model_from_json(model_json)
        # load weights
        #self.loaded_model.load_weights("trained.h5")
        self.loaded_model=keras.models.load_model("New_Age_sex_detection.h5")
    print("*******************LOADED*******")

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=40, size=48):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w,h) * margin / 100)
        x_a = x - margin
        y_a = y - margin
        x_b = x + w + margin
        y_b = y + h + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w-1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h-1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)
        
        
    def detect_face(self):
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)
        video_capture = cv2.VideoCapture('http://192.168.43.1:8080')
        # infinite loop, break by key ESC
        print("video start")
        while True:
            if not video_capture.isOpened():
                sleep(5)
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            #print("captured")    
         
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,minSize=(self.face_size, self.face_size))
            # placeholder for cropped faces
            face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
            for i, face in enumerate(faces):
                face_img, cropped = self.crop_face(frame, face, margin=40, size=self.face_size)
                (x, y, w, h) = cropped
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                face_imgs[i,:,:,:] = face_img/255
                
            
                
            #print("predicting")
            sex_f=['Male','Female']
            
            if len(face_imgs) > 0:
                # predict ages and genders of the detected faces
                pred_1 = self.loaded_model.predict(face_imgs)
                #predicted_genders = results[0]
                #sex_f=['Male','Female']
                age=int(np.round(pred_1[1][0]))
                
                
         
                sex=int(np.round(pred_1[0][0]))
                
            for i, face in enumerate(faces):
                label = "{}, {}".format(str(age),sex_f[sex])
                self.draw_label(frame, (face[0], face[1]), label)
                
            cv2.imshow('Keras Faces', frame)
            if cv2.waitKey(5) == 27:  # ESC key press
                break
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()
        
        
def main():


    face = FaceCV()
    face.detect_face()

if __name__ == "__main__":
    main()
        