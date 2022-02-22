import cv2
import numpy as np 

# Menjalankan Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
kelas = [] 
with open ("coco.names", "r") as coco:
    kelas = [line.strip() for line in coco.readlines()] # Mengambil Data dari coco.names untuk dimasukkan kedalam array
nama_layer = net.getLayerNames()
output_layer = [nama_layer[i[0]-1] for i in net.getUnconnectedOutLayers()]
warnapp = np.random.uniform(0,255, size=(len(kelas), 3)) # membuat warna berbeda - beda  pada persegi panjang Setiap kali di run program akan berubah warna

# Membaca gambar
img = cv2.imread("car.jpg") #File IMG
img = cv2.resize(img,None,fx=0.7,fy=0.7) #Ukurang Gambar
height,width, channels = img.shape

# Mendeteksi Objek
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0,0,0), True, crop=False)

net.setInput(blob)
keluar = net.forward(output_layer)

#Menapilkan Informasi
idkelass = []
confidences = []
box = []
for out in keluar:
    for deteksi in out:
        skor = deteksi[5:]
        idkelas = np.argmax(skor)
        confidence = skor[idkelas]
        confidence *= confidence
        if confidence > 0.5: # untuk memilhah objek yang terdeteksi 0.5 = 50% jadi yang akan di tampilkan objek yang mempunyai skor diatas 50%
            # Objek Terdeteksi
            center_x = int(deteksi[0] * width)
            center_y = int(deteksi[1] * height)
            w = int(deteksi[2] * width)
            h = int(deteksi[3] * height)

           #Persegi Panjang untuk Deteksi Objek
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            box.append([x,y,w,h])
            confidences.append(float(confidence))
            idkelass.append(idkelas)

index = cv2.dnn.NMSBoxes(box, confidences,0.5,0.4)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(box)):
    if i in index:
        x,y,w,h = box[i]
        label = kelas[idkelass[i]] #Untuk Memprint coco.names
        confidence = confidences[i]
        warna = warnapp[i]
        cv2.rectangle(img,(x,y),(x+w,y+h), warna,2)
        cv2.putText(img,label + " " + str(round(confidence*100,3))+"%",(x,y+30),font,1,(255,255,255),2)

#Menampilkan Gambar
cv2.imshow("Object Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows

