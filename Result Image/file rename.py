
# 遍历文件夹下所有文件
import os
g = os.walk('./')
for path,dir_list,file_list in g:
	print(path)
	print(dir_list)
	print(file_list)
	for file_name in file_list:
		print(os.path.join(path,file_name))
		newname=file_name.replace('epoch','Epoch')
		os.rename(file_name,newname)