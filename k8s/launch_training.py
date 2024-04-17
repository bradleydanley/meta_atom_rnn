import os
import sys
import yaml
import time
import subprocess
from IPython import embed

from dateutil.tz import tzutc
from kubernetes import client, config
from jinja2 import Environment, FileSystemLoader

from k8s_support import exit_handler, load_file, save_file, parse_args, load_config

sys.path.append("../")
from utils.general import create_folder

def run(params):

    template = load_file(params['kube']['path_template'])
    tag = params['kube']['path_template'].split("/")[-1]
    folder = params['kube']['path_template'].replace("/%s" % tag, "")
    environment = Environment(loader = FileSystemLoader(folder))
    template = environment.get_template(tag)

    create_folder(params['kube']['path_job_files'])

    #job_tags = params['visualize']['sequences'] * 2 # each sequence gets trained on rnn and lstm

    #counter = 0

    #current_group = []

    sequences = params['visualize']['sequences']
    arches = [0, 1]

    for sequence in sequences:

        params['dataset']['seq_len'] = sequence 

        for arch in arches:

            params['network']['arch'] = arch
            
            arch_str = 'rnn' if arch == 0 else 'lstm' 
            job_name = "%s-%s" % (arch_str, str(sequence).zfill(2))

            template_info = {'job_name': job_name,
                             'path_image': params['kube']['path_image'],
                             'path_logs': params['kube']['path_logs'],
                             'pvc_name': params['kube']['pvc_name'],
                             'sequence': params['dataset']['seq_len'],
                             'arch': params['network']['arch'], 
                            }

            filled_template = template.render(template_info)

            path_job = os.path.join(params['kube']['path_job_files'], job_name.zfill(2) + ".yaml")

            save_file(path_job, filled_template)

            subprocess.run(['kubectl', 'apply', '-f', path_job])
            print(f"launching job for {arch}, {sequence}")
         
    
if __name__=="__main__":

    kill = True
    args = parse_args(sys.argv)

    params = load_config(args['config'])

    if kill == False:
        run(params)
    elif kill == True:
        kill_tags = params['kube']['kill_tags']
        for kill_tag in kill_tags: 
            exit_handler(params,kill_tag)
