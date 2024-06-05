
#  Creating kubernetes jobs for loading results (params['experiment'] == 1) 
#  and running eval (params['experiment'] == 2)

import os
import sys
import yaml
import time
import subprocess

from dateutil.tz import tzutc
from kubernetes import client, config
from jinja2 import Environment, FileSystemLoader

from k8s_support import exit_handler, load_file, save_file, parse_args, load_config

sys.path.append("../")
from utils.general import create_folder

def run(params):

    job = 'load_results_job'

    job_name_origin = job.replace('_','-')    

    template = load_file(params['kube'][job]['paths']['template'])
    tag = params['kube'][job]['paths']['template'].split("/")[-1]
    folder = params['kube'][job]['paths']['template'].replace("/%s" % tag, "")
    environment = Environment(loader = FileSystemLoader(folder))
    template = environment.get_template(tag)

    create_folder(params['kube']['job_files'])
    
    sequences = params['visualize']['sequences']
    for sequence in sequences:
        params['dataset']['seq_len'] = sequence
    
        job_name = "%s-%s" % (job_name_origin, str(sequence))
 
        template_info = {'job_name': job_name,
                        'num_cpus': str(params['kube']['train_job']['num_cpus']),
                        'num_gpus': str(params['kube']['train_job']['num_gpus']),
                        'num_mem_req': str(params['kube']['load_results_job']['num_mem_req']),
                        'num_mem_lim': str(params['kube']['load_results_job']['num_mem_lim']),
                        'pvc_preprocessed': params['kube']['pvc_preprocessed'],
                        'pp_data_path': params['kube']['pp_job']['paths']['data']['preprocessed_data'],
                        'pvc_results': params['kube']['pvc_results'],
                        'results_path': params['kube']['train_job']['paths']['results']['model_results'],
                        'ckpt_path': params['kube']['train_job']['paths']['results']['model_checkpoints'],
                        'path_image': params['kube']['image'],
                        #'path_logs': params['kube']['path_logs'],
                        'sequence': sequence, 
                        }   
        
        filled_template = template.render(template_info)

        path_job = os.path.join(params['kube']['job_files'], job_name.zfill(2) + ".yaml")

        save_file(path_job, filled_template)

        subprocess.run(['kubectl', 'apply', '-f', path_job])
        print(f"launching job for {job_name}")
        #from IPython import embed; embed();





         
    
if __name__=="__main__":


    kill = False
    #kill = True
     
    args = parse_args(sys.argv)

    params = load_config(args['config'])

    if kill == False:

        run(params)

    elif kill == True:
        kill_tag = params['kube']['load_results_job']['kill_tag']
        exit_handler(params,kill_tag)

  
