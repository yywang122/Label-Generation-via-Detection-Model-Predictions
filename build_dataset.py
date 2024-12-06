import os
import random
import math
import shutil
import argparse


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Building Yolo Dataset", add_help=add_help)
    parser.add_argument("--ori_path", default="",type=str, help="path to initial dataset containing both label and image files")
    parser.add_argument("--save_path", default="",type=str, help="path to save the rebuilt dataset")
    parser.add_argument("--train", type=float, default="0.7", help="partition of training data in the dataset")
    parser.add_argument("--val", type=float, default="0.2", help="partition of validation data in the dataset")
    return parser


def copy_files(src_dir,files_li,dst_dir):
    for f in files_li:
        shutil.copyfile(os.path.join(src_dir,f),os.path.join(dst_dir,f))
        #break


def copy_files_bypath(src_path_li,dst_dir):
    for p in src_path_li:
        shutil.copyfile(p,os.path.join(dst_dir,os.path.basename(p)))
        #break


def pair_img_label_files(dir_paths_list):
    for dir_path in dir_paths_list:            
        for r ,d ,f in os.walk(dir_path):            
            for i_f in f:
                f_name = i_f[:-4]
                label_path = os.path.join(r,f_name+'.txt')
                img_path = os.path.join(r,f_name+'.jpg')
                label_extist = os.path.exists(label_path)
                img_extist = os.path.exists(img_path)
                #print(label_extist)
                #print(img_extist)

                
                if label_extist and img_extist:
                    pass
                else:
                    if not img_extist:
                        cmd_delete_file = f'rm -r "{img_path}"'
                        #print(cmd_delete_file)
                        os.system(cmd_delete_file)
                    if not label_extist:
                        cmd_delete_file = f'rm -r "{label_path}"'
                        #print(cmd_delete_file)
                        os.system(cmd_delete_file)



def rename_repeat_files(ori_path):
    label_files_paths=[]
    img_files_paths=[]
    files_names=[]    

    for r ,d ,f in os.walk(ori_path):
        for i_f in f:
            if '.txt' in i_f:
                label_ori_path = os.path.join(r,i_f)
                if i_f[:-4]  not in files_names:          
                    files_names.append(i_f[:-4])
                    label_files_paths.append(label_ori_path)
                    img_files_paths.append(label_ori_path.replace('.txt','.jpg'))                 

                else:                     
                    img_ori_path=label_ori_path.replace('.txt','.jpg')
                    d_name=r.split('/')[-1]
                    new_i_f = i_f[:-4]+'_'+d_name           
                    label_new_path = os.path.join(r,new_i_f+'.txt')
                    img_new_path = os.path.join(r,new_i_f+'.jpg')

                    cmd_label_rename = f"mv '{label_ori_path}' '{label_new_path}'"
                    cmd_img_rename = f"mv '{img_ori_path}' '{img_new_path}'"
                    os.system(cmd_label_rename)
                    os.system(cmd_img_rename)

                    files_names.append(new_i_f)
                    label_files_paths.append(label_new_path)
                    img_files_paths.append(img_new_path)
    return label_files_paths,img_files_paths

   

def main(args):
    print('hihi')    
    ori_path= args.ori_path
    dataset = args.save_path

    os.makedirs(dataset,exist_ok=True)
    os.makedirs(dataset+'/train',exist_ok=True)
    os.makedirs(dataset+'/train/images',exist_ok=True)
    os.makedirs(dataset+'/train/labels',exist_ok=True)
    os.makedirs(dataset+'/valid',exist_ok=True)
    os.makedirs(dataset+'/valid/images',exist_ok=True)
    os.makedirs(dataset+'/valid/labels',exist_ok=True)
    os.makedirs(dataset+'/test',exist_ok=True)
    os.makedirs(dataset+'/test/images',exist_ok=True)
    os.makedirs(dataset+'/test/labels',exist_ok=True)


    dir_list =sorted(os.listdir(ori_path))
    dir_paths_list=[os.path.join(ori_path,d) for d in dir_list]
    # print(dir_paths_list)

    #Check and Remove Unpaired Label and Image Files
    pair_img_label_files(dir_paths_list)

    #Rename Duplicate Files to Prevent Conflicts 
    label_files_paths,img_files_paths=rename_repeat_files(ori_path) 

    assert len(label_files_paths)==len(img_files_paths),' label files num  is not equal to img files num'        
    assert [os.path.basename(i)[:-4] for i in img_files_paths]==[os.path.basename(l)[:-4] for l in label_files_paths], 'image filesnames are not all matched to label filenames.'

    index_li=[i for i in range(len(label_files_paths))]
    random.seed(0)
    random.shuffle (index_li )

    train_num= math.ceil(len(label_files_paths)*args.train)
    val_num= math.ceil(len(label_files_paths)*args.val)
    test_num= len(label_files)-train_num-val_num

    train_index = index_li[0:train_num]
    #val_index = index_li[train_num:]
    val_index = index_li[train_num:train_num+val_num]
    test_index = index_li[-test_num:]


    # print(train_index,len(train_index))
    # print(val_index,len(val_index))
    # print(test_index,len(test_index))



    train_img = list(map(img_files_paths.__getitem__, train_index))
    val_img = list(map(img_files_paths.__getitem__, val_index))
    test_img = list(map(img_files_paths.__getitem__, test_index))


    train_label = list(map(label_files_paths.__getitem__, train_index))
    val_label = list(map(label_files_paths.__getitem__, val_index))
    test_label = list(map(label_files_paths.__getitem__, test_index))

    assert [os.path.basename(i)[:-4] for i in train_img]==[os.path.basename(l)[:-4] for l in train_label], 'Train: image files are not all matched to label_files.'
    assert [os.path.basename(i)[:-4] for i in val_img]==[os.path.basename(l)[:-4] for l in val_label], 'Val: image files are not all matched to label_files.'
    assert [os.path.basename(i)[:-4] for i in test_img]==[os.path.basename(l)[:-4] for l in test_label], 'Val: image files are not all matched to label_files.'



    copy_files_bypath(train_label,dataset+'/train/labels')
    copy_files_bypath(train_img,dataset+'/train/images')
    copy_files_bypath(test_img,dataset+'/test/images')

    copy_files_bypath(val_label,dataset+'/valid/labels')
    copy_files_bypath(val_img,dataset+'/valid/images')
    copy_files_bypath(test_img,dataset+'/test/images')



if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    print('initial_args: ',args)
    main(args)