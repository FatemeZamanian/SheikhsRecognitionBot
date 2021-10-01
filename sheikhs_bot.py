from keras.models import load_model
import telebot
from telebot import types
# import numpy as np
import cv2
import os

token = os.environ["token"]
bot = telebot.TeleBot(token)

model=load_model('sheikhs.h5')

# btns=telebot.types.ReplyKeyboardMarkup(row_width=1)
# btn1=telebot.types.KeyboardButton('Start')
# btns.add(btn1)
@bot.message_handler(commands=['start'])
def wlc(message):
    bot.send_message(message.chat.id,'Hi 👋🏻 please send me a picture👀',
    #reply_markup=btns
     )

@bot.message_handler(content_types=['photo'])
def photo(message):
    fileID = message.photo[-1].file_id
    file_info = bot.get_file(fileID)
    src=file_info.file_path
    downloaded_file = bot.download_file(src)
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)

    img_org = cv2.imread(src)
    img_RGB = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_RGB, (224, 224))
    
    img_numpy = numpy.array(img_resize)
    img = img_numpy / 255.0
    final = img.reshape(1, 224, 224, 3)

    y_pred = numpy.argmax(model.predict(final))

    if y_pred == 0:
        bot.reply_to(message,'normal person')
      
    else:
        bot.reply_to(message,'sheikh')
      

bot.polling()
