import os

main_folder = "/data2/nlazaridis/sal_dataset/DHF1K/test_set/results"
counter=0
for subdir in os.listdir(main_folder):
    subdir_path = os.path.join(main_folder, subdir)
    for filename in os.listdir(subdir_path):
        if filename.endswith(".png"):
            #old_name = os.path.join(subdir_path, filename)
            #new_name = os.path.join(subdir_path, str(int(filename[:-4])+1).zfill(4) + ".png")
            #os.rename(old_name, new_name)
            counter = counter + 1

print(counter)