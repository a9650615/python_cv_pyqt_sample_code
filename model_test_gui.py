import time
from subprocess import Popen, check_output, check_call, PIPE, call

your_exe_file_address = "./Mnist" # example
your_command = 'mnist_model.tflite'
your_module_address = "five.bin" # example

process = Popen([your_exe_file_address], stdout=PIPE, stdin=PIPE, stderr=PIPE, shell=True) #, shell=True
# stdout, stderr = process.communicate()

getRes = False

# output = process.communicate(b'five.bin')[0].decode().strip()
# print('out', output)
process.poll()
# print('out', out.strip())
process.stdin.write(b"five.bin\r\n")
# process.stdin.close()
process.stdin.flush()
# process.stdin.close()
# data,ww = process.communicate(b"five.bin")
# print(data)
# print(ww)
output = process.stdout.readline()
print('out', output.strip())
while True:
	output = process.stdout.readline()
	if output == '' and process.poll() is not None:
		break
	if output:
		if output.strip().find(b"Result") != -1:
			getRes = True

		print('out2', output.strip())
	if getRes == True:
		# process.communicate(b"five.bin\n")
		process.stdin.write(b'five.bin\r\n')
		process.stdin.flush()
	# process.stdin.close()
	
	rc = process.poll()
	time.sleep(0.02)

print("finish")
# print(stdout, stderr)
