from flask import Flask,render_template,request
import os
from random import random
import cv2
from ultralytics import YOLO

model = YOLO('weights/checkpoint.pt')

# Tạo một app bằng Flask
app = Flask(__name__)
# khi client uploadfile thì ảnh sẽ được lưu vào static
app.config["UPLOAD_FOLDER"] = "static"

@app.route('/',methods=['GET','POST'])

def homepage():
    if request.method == "POST":
        try:
            image = request.files["file"]
            print(request.files)

            if image:
                # lưu đường dẫn đến file client upload
                path_to_save = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
                print("Save in " , path_to_save)
                image.save(path_to_save)
                # đọc ảnh và gán vào biến frame
                frame = cv2.imread(path_to_save)
                # đưa vào model
                results= model(frame)
                results_img = results[0].plot()
                # ghi đè ảnh sau khi được bounding box vào static
                cv2.imwrite(path_to_save, results_img)
                return render_template("index.html", user_image = image.filename , rand = str(random()),
                                           msg="Tải file lên thành công", ndet = len(results[0].boxes))
            else:
                 return render_template('index.html', msg='Hãy chọn file để tải lên')
        except Exception as e:
            print(e)
            return render_template('index.html', msg='Có lỗi xảy ra khi tải file lên')
    # nếu là GET
    else:
        return render_template('index.html')
if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)

