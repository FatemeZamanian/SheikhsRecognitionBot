
import telebot
import numpy
import cv2
import os
from tensorflow.keras.models import load_model

token = os.environ["token"]
bot = telebot.TeleBot(token)


model=load_model('sheikhs.h5')

@bot.message_handler(commands=['start'])
def wlc(message):
    bot.send_message(message.chat.id,'Hi 👋🏻 please send me a picture👀')

@bot.message_handler(content_types=['photo'])
def send_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    image_path = os.path.join('input', file_info.file_path)
    with open(image_path, 'wb') as new_file:
        new_file.write(downloaded_file)
        
    img_org = cv2.imread(image_path)
    img_RGB = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_RGB, (224, 224))
    
    img_numpy = numpy.array(img_resize)
    img = img_numpy / 255.0
    final = img.reshape(1, 224, 224, 3)

    y_pred = numpy.argmax(model.predict(final))
    print(y_pred)
    if y_pred == 0:
        bot.reply_to(message,'normal person')
      
    elif y_pred==1:
        bot.reply_to(message,'sheikh')
      

bot.polling()
