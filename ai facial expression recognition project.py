#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[6]:


from deepface import DeepFace


# In[15]:


img=cv2.imread('happyboy2.jpg')


# In[16]:


import matplotlib.pyplot as plt


# In[17]:


plt.imshow(img)


# In[18]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[21]:


predictions=DeepFace.analyze(img)


# In[22]:


predictions


# In[23]:


type(predictions)


# In[24]:


predictions['dominant_emotion']


# In[25]:


faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


# In[26]:


gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray,1.1,4)

#draw a rectangle around the face
for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


# In[27]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[36]:


font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,
            predictions['dominant_emotion'],
           (0,50),
           font,3,
           (0,0,255),
           2,
           cv2.LINE_4);


# In[37]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[38]:


img=cv2.imread('angryboy.jpg')


# In[39]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[40]:


predictions=DeepFace.analyze(img)


# In[41]:


predictions


# In[46]:


faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


# In[47]:


gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray,1.1,4)

#draw a rectangle around the face
for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


# In[48]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[61]:


import cv2
from deepface import DeepFace
faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(1)
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open webcam")
while True:
    ret,frame=cap.read()
    result=DeepFace.analyze(frame,actions=['emotion'])
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,
                result['dominant_emotion'],
                (50,50),
                font,3,
                (0,0,255),
                2,
                cv2.LINE_4)
    cv2.imshow('Original video',frame)
    if(cv2.waitKey(2)&0xFF==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
    


# In[51]:


faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


# In[54]:


img=cv2.imread('sadwomen.jpg')


# In[55]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[56]:


faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


# In[57]:


gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray,1.1,4)

#draw a rectangle around the face
for(x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


# In[58]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[59]:


font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,
            predictions['dominant_emotion'],
           (0,50),
           font,3,
           (0,0,255),
           2,
           cv2.LINE_4);


# In[60]:


plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))


# In[ ]:




