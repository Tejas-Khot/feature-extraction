package.path = package.path .. ";../?.lua"

require 'loadcaffe'
require 'xlua'
require 'optim'
require 'image'
require 'hdf5';

require 'neuralfeature'

-- prepare data

prototxt='/home/profloo/Downloads/Caffe_Models/VGG_ILSVRC_16_layers_deploy.prototxt.txt'
binary='/home/profloo/Downloads/Caffe_Models/VGG_ILSVRC_16_layers.caffemodel'

path="/home/profloo/Desktop/Train_SFEW_Features/namelist/"

local p = io.popen('find "'..path..'" -type f')  
local dirlist = {}
local i=1
   for file in p:lines() do                          
	dirlist[i]=file
	i=i+1   
   end
print(dirlist)

-- load as net
net = loadcaffe.load(prototxt, binary)
-- switch off dropout
net:evaluate();

function save_file(file_adr, group_adr, out_save)
    local my_file=hdf5.open(file_adr, 'w');
    my_file:write(group_adr, out_save);
    my_file:close()
end


for i = 1, #dirlist do
	images = neuralfeature.loadimagelist(dirlist[i])
	out, labels=neuralfeature.extract(net, images)
	print(string.sub(dirlist[i],52))
	out_save=torch.zeros(#images, 1000)

	for i=1, #images do
	    out_save[{ i, {} }]=out[i]    
	end

	print(out_save:size())

	save_file("/home/profloo/Desktop/Train_SFEW_Features/" .. string.sub(dirlist[i],52) .. ".h5",
			   string.sub(dirlist[i],52),
			   out_save)
	-- myFile=hdf5.open("/home/profloo/Desktop/Train_SFEW_Features/" .. string.sub(dirlist[i],52) .. ".h5", 'w');
	-- myFile:write("/home/profloo/Desktop/Train_SFEW_Features/" .. string.sub(dirlist[i],52) .. ".h5", out_save);
	-- myFile:close()
	collectgarbage()
end
print("Done!")

