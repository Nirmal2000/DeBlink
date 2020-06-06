from flask import Flask
from flask_socketio import SocketIO,send

app = Flask(__name__)
app.config['SECRECT_KEY'] = '12345678'
socketio = SocketIO(app)

@socketio.on('message')
def handleMessage(msg):
    print("Message :"+msg)
    send(msg,broadcast=True)

@app.route('/')
def hello_world():
    return "A"

if __name__ == '__main__':
    socketio.run(app,debug=True)
    