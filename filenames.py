import os

path="/home/profloo/Downloads/Train_SFEW_2_0/"
write_path="/home/profloo/Desktop/Train_SFEW_Features/"
dirs=os.listdir(path)
for dir in dirs:
	paths = [os.path.join(path+dir,fn) for fn in next(os.walk(path+dir))[2]]
	f=open(write_path+dir,"w")
	f.write("\n".join(paths))
	f.close()
	
