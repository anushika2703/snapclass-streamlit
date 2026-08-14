import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st

from src.database.db import get_all_students

@st.cache_resource
def load_dlib_models():
    detector=dlib.get_frontal_face_detector() #jo bounding box hai na, box detector voh dlib de dega

    #yeh landmark dega face pr; iske liye humne github id use kri hai
    sp=dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    facerec=dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec

#face embeddings generate krne ke liye
def get_face_embeddings(image_np):
   detector,sp,facerec=load_dlib_models()
   faces=detector(image_np,1)#humne 1 rkha hai kyunki yeh face ko rotate flip krke dekhega (preprocess) and if we insrease the count ofc accuracy will increase but we don't want to give our small scale CPU or OS a hard time.
   encodings=[]
   for face in faces:
       shape=sp(image_np,face)
       face_descriptor=facerec.compute_face_descriptor(image_np,shape,1) #128 embeddings 


       #ab hum embeddings ko numpy mein convert krenge for easy further process
       encodings.append(np.array(face_descriptor))
   return encodings

#encodings- image jo hai abb number ke format mein convert ho chuki hai
@st.cache_resource
def get_trained_models():
    X=[]
    y=[]
    student_db=get_all_students()
    if not student_db:
        return None
    for student in student_db:
        embedding=student.get('face_embedding')
        if embedding:
            X.append(np.array(embedding))
            y.append(student.get('student_id'))
    if len(X)==0:
        return 0
    
    #building classifier; straight line divison
    clf=SVC(kernel='linear',probability=True, class_weight='balanced')
    try:
        clf.fit(X,y)
    except ValueError:
        pass 

    return {'clf':clf,'X':X,'y':y}


def train_classifier():
    #database mein new formula lgana hai
    st.cache_resource.clear()
    model_data = get_trained_models()
    return bool(model_data)

#attendance ko predict kr dega, classroom ki photo le lega array ke form mein
def predict_attendance(class_image_np):
    #sari face embeddings nikal lenge uss group photo se 
    encodings=get_face_embeddings(class_image_np)

    detected_student={}

    model_data=get_trained_models() 
    if not model_data:
        return detected_student,[], len(encodings)
    
    clf=model_data['clf']
    X_train=model_data['X']
    y_train=model_data['y']
    
    #set dalkr mai sirf unique students ko call krr rhi hu taki koi student do baar na aa jaye
    all_students = sorted(list(set(y_train)))


#abb hum sare detected students ko detected_Student mein dalenge
    for encoding in encodings:
       if len(all_students)>= 2:
           predicted_id=int(clf.predict([encoding])[0])
       else:
           predicted_id=int(all_students[0])

       student_embedding=X_train[y_train.index(predicted_id)]
       #classifier ne toh keh diya ki same hai pr hum mathematically bhi verify krenge ki same hai chehra ya nhi
  #humne ek threshold set kiya hai 
       best_match_score = np.linalg.norm(student_embedding - encoding)
       resemblance_threshold=0.6
       if best_match_score<=resemblance_threshold:
           detected_student[predicted_id]=True
    return detected_student, all_students, len(encodings)
           


       
