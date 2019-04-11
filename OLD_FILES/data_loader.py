from threading import Thread
import time

class data_loader():

	def __init__(self,transform_map,target_size,batch_size,max_queue_size):
		self.transform_map=transform_map
		self.target_size=target_size
		self.batch_size=batch_size
		self.max_queue_size=max_queue_size
		self.queue=[]
		self._terminate=False
		self._pause=False

	def pop(self):
		if len(self.queue)<1:
			print('list empty, generating data')
			return False
		return self.queue.pop()

	def generate_data(self):
		print('generating data')
		if not self._pause:
			self.queue.append(dataGen.image_processor_batch(transform_map=self.transform_map,target_size=self.target_size,batch_size=self.batch_size))

	def terminate(self):
		self._terminate=True
		Thread.join()

	def update_batch_size(self,batch_size):
		self._pause=True
		time.sleep(3)
		self.batch_size=batch_size
		self.queue=[]
		self._pause=False

	def get_queue_size(self):
		return len(self.queue)

	def get_max_queue_size(self):
		return self.max_queue_size

	def get_terminate(self):
		return self.terminate

def data_loader_generator(data_loader):
	max_queue_size=data_loader.get_max_queue_size()
	while data_loader.get_queue_size()<max_queue_size:
		if data_loader.get_terminate():break
		data_loader.generate_data()
		print('data generated')
