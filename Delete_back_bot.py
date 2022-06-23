import telebot
import cv2
from io import BytesIO
import numpy as np
from Delete_background import Delete_back
from config import TOKEN

bot = telebot.TeleBot(TOKEN)

# Load photo from message -> cv2 image array
def Photo_to_Image(message):
    fileID = message.photo[-1].file_id
    file_path = bot.get_file(fileID).file_path
    downloaded_file = bot.download_file(file_path)

    img_bytes = np.asarray(bytearray(downloaded_file), dtype=np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    return img

def Image_to_Bytes(img):
    is_success, byte_img = cv2.imencode(".jpg", img)
    new_img = BytesIO(byte_img).getvalue()
    return new_img

@bot.message_handler(content_types=['photo'])    
def image_handler(message):
    img = Photo_to_Image(message)
    new_img = Delete_back(img)
    new_img = Image_to_Bytes(new_img)
    bot.send_photo(message.chat.id, new_img)

@bot.message_handler(commands=['start'])
def Greetings(message):
    greet_mes = 'Hi, please, send me your picture and I cut its background for you'
    bot.send_message(message.chat.id, greet_mes)

bot.polling(none_stop=True)
