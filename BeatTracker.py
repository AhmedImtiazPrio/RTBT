import os
import numpy as np
import time
from scipy.io import wavfile
from src import *
import array
import wave
import pyaudio
# import serial

fs=44100

#############################################################
#Declare Beat Storing Variable (will be cast into numpy array)
beats=[]

######################
#PREDICTION PARAMETERS

#First calculation after 4s
start_time = 4 

#Maximum amount of data (in sec) evaluated each time for beat tracking
chunk_len = 30

#Hop (in sec) between subsequent window centers
hop_size = 1

#Prediction window (in sec)
pred_factor=1.1
pred_window = pred_factor*hop_size

#Rejection threshold for making decision on accepting a future beat 
rejection_thres=0.35


##############################################
#TEMPO ESTIMATION and BEAT TRACKING PARAMETERS

#Tempo estimation's (in bpm) centre of log-gaussian window
start_bpm=210

#Controls deviation of tempo from estimated tempo duting beat tracking 
tightness=95 

#Hop (in samples) between subsequent window centers for onset envelope
hop_length=1024 

#Window length (in samples) for onset envelope
n_fft=4096

#####################
#ALIGNMENT PARAMETERS

#Extra subsequently rejected data evaluated during each onset calculation
lagg=int(round(n_fft/(hop_length)))

#Last time (in sec) to be evaluated
end_time = 30*44100/fs - pred_window + hop_size

#List of time (in sec) when calculation takes place
chunk = np.arange(start_time+hop_size, end_time, hop_size)
ftime = chunk[-1]*fs+hop_size*fs
data = []
env = []
i=0

class BeatTracker:
	"""docstring for Beat
Tracker"""
	def __init__(self, outdata = 'terminal', serialName = 'None'):

		self.outdata = outdata
		self.mySerial = serialName

	def Beats(self,InFile = 'sample.wav', OutFile = 'sample.txt'):
		global fs, beats,start_time,chunk_len,hop_size,pred_factor,pred_window,rejection_thres,start_bpm
		global	tightness,hop_length,n_fft,lagg,end_time,chunk,ftime,data,env,i 
		beats = []
		pred_window = 1.1
		data = []
		flag = 0
		env = []

		fs,data=wavfile.read(InFile,44100)

		###########################################################
		#Parameters to be changed for songs shorter/longer than 30s
		
		#Last time (in sec) to be evaluated
		end_time = len(data)/fs - pred_window + hop_size
		
		#List of time (in sec) when calculation takes place
		chunk = np.arange(start_time+hop_size, end_time, hop_size)

		######################################################
		#First 4s evaluation generates pred_window future beats
		
		self.BeatTrackerInit()

		#############################################################################
		#Evaluation of 4s to end of song each time generating pred_window future beats
		
		for i in range(0,len(chunk)):
			self.BeatTracker()
		################################
		# Save time stamps to a file#
		myfile=open(OutFile,'w+')
		for item in beats:
			myfile.write("%s\n" %item)
		myfile.close()
		return beats

	def BeatTrackerInit(self):
		global data, beats, env 

		#Select data chunk with rejection window at the end
		ys=data[0:int(start_time*fs)+lagg*hop_length]

		#Onset envelope
		env,SS = onset_strength(y=ys, sr=fs,  hop_length=hop_length,
								aggregate=np.mean, center=False, n_fft=n_fft)
		
		#Beat tracking (in frames) using above onset
		s,beatss,SS=beat_track(ys, onset_envelope=env, 
								sr=fs,start_bpm=start_bpm,
								tightness=tightness,hop_length=hop_length)
		
		#Convert frame number to time to get beat position (in sec)
		b_new=frames_to_time(beatss, sr=fs,hop_length=hop_length)
		
		#Total deviation
		tempo = sum(b_new[1:] - b_new[0:-1])
		#i=start_time
		
		#When more than one beat found in 4s
		if len(b_new)>1:
			
			#Set tempo as average tempo from current beats
			tempo /= (len(b_new) - 1)
			
			#Add starting point of current chunk
			b_new=b_new+max(0, (start_time - chunk_len))
			
			#Next beat position
			new = b_new[-1] + tempo
			
			while (1):
				
				#End prediction at edge of predition window
				if (new >= start_time + pred_window or new>=30):
					break
					
				#Append future beat positions
				if ((new-b_new[-1])/((b_new[-1]-b_new[-2])*1.0)>rejection_thres):
					#write only future beats into output
					if (new>start_time+(pred_factor-1)/2.0):
						beats = np.append(beats, new)
					b_new=np.append(b_new,new)
				
				#Next beat position
				new=new+tempo
				
		#Handle outlier condition when just one beat found
		else:
			
			#Assuming first beat sets tempo in this condition
			tempo = b_new
			
			#Add starting point of current chunk
			b_new=b_new+max(0, (start_time - chunk_len))
			
			#Next beat position
			new = b_new[-1] + tempo
			
			while (1):
				
				#End prediction at edge of predition window
				if (new >= start_time + pred_window or new>=30):
					break
				
				#Append future beat positions when deviation beyond threshold
				if (len(beats)>1):
					if ((new-b_new[-1])/((b_new[-1]-b_new[-2])*1.0)>rejection_thres):
						#write only future beats into output
						if (new>start_time+(pred_factor-1)/2.0):
							beats = np.append(beats, new)
						b_new=np.append(b_new,new)
				else:
					#write only future beats into output
					if (new>start_time+(pred_factor-1)/2.0):
						beats=np.append(beats,new)
					b_new=np.append(b_new,new)

				#Next beat position
				new=new+tempo	
	

	def BeatTracker(self):
		global data, sr, beats, i, env 

		#Select only hop_size (in sec) data chunk with rejection window at both ends
		ys=data[(int((chunk[i]-hop_size)*fs)//hop_length+1-lagg)*hop_length:int(chunk[i]*fs)+lagg*hop_length]

		#Onset envelope of selected window
		temp_env,SS = onset_strength(y=ys,
									sr=fs,
									hop_length=hop_length,
									aggregate=np.mean,center=False,n_fft=n_fft)	
		
		#Merge current onset with previous chunk's onset to get entire onset
		env=np.concatenate((env[:-1*lagg+1],temp_env[lagg+1:]),axis=0)
		
		#Onset chunk of size chunk_len (in sec) used to track beats
		env2=env[int((max(0, (chunk[i] - chunk_len))*fs)//hop_length):]
		
		#Beat tracking (in frames) using above onset
		s,b_neww,SS=beat_track(data[0:int(chunk[i]*fs)+lagg*hop_length],
								onset_envelope=env2,sr=fs,
								start_bpm=start_bpm,tightness=tightness,
								hop_length=hop_length)
								
		#Convert frame number to time to get beat position (in sec)
		b_new=frames_to_time(b_neww, sr=fs,hop_length=hop_length)
		
		#Reject outlier condition when just one beat is found
		if len(b_new) <= 1:
			return
			
		#Set tempo as average tempo from current beats
		tempo = sum(b_new[1:] - b_new[0:-1])
		tempo /= (len(b_new) - 1)
		
		#Add starting point of current chunk
		b_new=b_new+max(0, (chunk[i] - chunk_len))
		
		#Next beat position
		new = b_new[-1] + tempo
		
		while (1):
			
			#End prediction at edge of predition window
			if (new >= chunk[i] + pred_window or new>=30):
				break
				
			#Append future beat positions when deviation beyond threshold
			if (len(beats)>1):
				if((new-beats[-1])/((beats[-1]-beats[-2])*1.0)>rejection_thres):
					beats = np.append(beats, new)
			else:
				beats=np.append(beats,new)
			new=new+tempo
		i+=1
			

	def MicrophoneStream(self):
		CHANNELS = 1
		RATE = 44100
		global fs, beats,start_time,chunk_len,hop_size,pred_factor,pred_window,rejection_thres,start_bpm
		global	tightness,hop_length,n_fft,lagg,end_time,chunk,ftime,data,env,i 
		beats = []
		pred_window = 1.5
		data = []
		flag = 0
		env = []
		i = 0
		beatCount = 0
		
		p = pyaudio.PyAudio()
		#callback function for non blocking audio acquisition 
		def callback(in_data, frame_count, time_info, flag):
			global data 
			buff = np.fromstring(in_data, dtype=np.float32)
			data = np.append(data,buff)
			return (buff, pyaudio.paContinue)
	
		try:
			raw_input('Press Enter To Start Streaming: ')
			## Open a Stream with 44.1k sampling rate in non blocking mode.
			stream = p.open(format=pyaudio.paFloat32,
						channels=CHANNELS,
						rate=RATE,
						output=False,
						input=True,
						stream_callback=callback)
			start_t=time.time()
			stream.start_stream()

			while stream.is_active():
				#Evaluation of First 4s generates pred_window future beats
				if flag == 0:
					if len(data) >= start_time*fs+lagg*hop_length:
						prevlen = len(data)
						self.BeatTrackerInit()
						flag = 1
				#Evaluation after 4s generates pred_window future beats
				else:
					mylen = len(data)
					if mylen - prevlen >= int(hop_size*fs) and mylen < ftime:
						prevlen = len(data)
						self.BeatTracker()
					if self.outdata == 'Serial':
						while len(beats) > beatCount:
							mySerial.write(str(beats[beatCount]))
							beatCount+=1
					elif self.outdata == 'terminal':
						if any(np.array(beats[beatCount:]) * 44100 <= mylen) and len(beats) > beatCount:
							print 'beat at  '+ str(beats[beatCount]) + 's'
							beatCount+=1
				#Keep stream open upto 30s 
				if len(data) >= 30*fs:
					print 'Enough for this session'
					break
			#Stop stream and terminate pyaudio module
			stream.stop_stream()
			stream.close()
			p.terminate()
		except KeyboardInterrupt:
			print '\n Audio Acuisition Finished'

	
	def wavStream(self, InFile='sample.wav', OutFile = 'sample.txt'):

		global fs, beats,start_time,chunk_len,hop_size,pred_factor,pred_window,rejection_thres,start_bpm
		global	tightness,hop_length,n_fft,lagg,end_time,chunk,ftime,data,env,i 
		beats = []
		pred_window = 1.1
		data = []
		flag = 0
		env = []
		i = 0
		beatCount = 0
		wf=wave.open(InFile,'rb')
		
		## Audio Streaming with pyaudio ##
		p = pyaudio.PyAudio()
		#CallBack function for Non blocking audio acquisition
		def callback(in_data, frame_count,time_info,status):
			global data
			buff = array.array('h',wf.readframes(frame_count))
			data = np.append(data, buff)
			return (buff, pyaudio.paContinue)

		try:
			raw_input('Press Enter To Start Streaming: ')
			
			## Open a Stream with 44.1k sampling rate in non blocking mode with 2048 frames per buffer
			stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
							channels = 1, rate = wf.getframerate(),
							output = True,stream_callback= callback,frames_per_buffer = 2048)
			start_t=time.time()
			# start streaming 
			stream.start_stream()
			# Start Processing
			while stream.is_active():
				# First 4s Evaluation generates pred_window future beats
				if flag == 0:
					if len(data) >= start_time*fs+lagg*hop_length:
						prevlen = len(data)
						self.BeatTrackerInit()
						flag = 1
				#Evaluation after 4s generates pred_window future beats
				else:
					mylen = len(data)
					if mylen - prevlen >= int(hop_size*fs) and mylen < ftime:
						prevlen = len(data)
						self.BeatTracker()
					if self.outdata == 'Serial':
						while len(beats) > beatCount:
							mySerial.write(str(beats[beatCount]))
							beatCount+=1
					elif self.outdata == 'terminal':
						if any(np.array(beats[beatCount:]) * 44100 <= mylen) and len(beats) > beatCount:
							print 'beat at  '+ str(beats[beatCount]) + 's'
							beatCount+=1
				# if len(data) >= 30*fs:
				# 	break
			########################################
			##### Write beats in a text file ######
			myfile = open(OutFile,'w+')
			for item in beats:
				myfile.write("%s\n" %item)
			myfile.close()
			#stop streaming and close pyaudio module
			stream.stop_stream()
			stream.close()
			p.terminate()
		except KeyboardInterrupt:
			print '\n Audio acquisition Finished'