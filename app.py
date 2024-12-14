import argparse
import json

from flask import Flask
from flask import render_template
from flask_socketio import SocketIO


class Visualizer:

    def __init__(self, simulation_path):
        self.simulation_path = simulation_path
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app,
                                 json=json,
                                 cors_allowed_origins="*",
                                 async_mode='eventlet')
        self.setup_routes()
        self.setup_socket_handlers()

    def setup_routes(self):

        @self.app.route('/')
        def index():
            return render_template('index.html')

    def setup_socket_handlers(self):

        @self.socketio.on('connect')
        def handle_connect():
            print('Client connected')

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('Client disconnected')

        @self.socketio.on('get_model')
        def handle_get_model():
            try:
                with open(self.simulation_path, 'r') as file:
                    data = json.load(file)
                self.socketio.emit('model', data['model'])
            except Exception as e:
                print(f"Error loading model: {e}")
                self.socketio.emit('error', {'message': 'Error loading model'})

        @self.socketio.on('get_states')
        def handle_get_states():
            try:
                with open(self.simulation_path, 'r') as file:
                    data = json.load(file)
                self.socketio.emit('states', data['states'])
            except Exception as e:
                print(f"Error loading states: {e}")
                self.socketio.emit('error',
                                   {'message': 'Error loading states'})

    def run(self, debug=True, host='0.0.0.0', port=5000):
        self.socketio.run(self.app, debug=debug, host=host, port=port)


def run_visualizer():
    parser = argparse.ArgumentParser()
    parser.add_argument('simulation_file',
                        type=str,
                        help='Simulation file path')
    args = parser.parse_args()

    visualizer = Visualizer(args.simulation_file)
    visualizer.run()


if __name__ == '__main__':
    run_visualizer()
