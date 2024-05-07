#  Creating kubernetes job for preprocessing data

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

   
    template = load_file(params['kube']['pp_job']['paths']['template'])
    
    tag = params['kube']['pp_job']['paths']['template'].split("/")[-1]
    folder = params['kube']['pp_job']['paths']['template'].replace("/%s" % tag, "")
    environment = Environment(loader = FileSystemLoader(folder))
    template = environment.get_template(tag)

    job_name = "%s" % (params['kube']['pp_job']['kill_tag'])

    template_info = {'job_name' : job_name,
                     'num_cpus' : str(params['kube']['pp_job']['num_cpus']),
                     'num_mem_lim' : str(params['kube']['pp_job']['num_mem_lim']),
                     'num_mem_req' : str(params['kube']['pp_job']['num_mem_req']),
                     'pvc_volumes' : params['kube']['pvc_volumes'],
                     'volumes_path' : params['kube']['pp_job']['paths']['data']['volumes'],
                     'pvc_preprocessed' : params['kube']['pvc_preprocessed'],
                     'preprocessed_path' : params['kube']['pp_job']['paths']['data']['preprocessed_data'],
                     'path_image' : params['kube']['image'],
                    }

    filled_template = template.render(template_info)

    #from IPython import embed; embed(); exit()
    path_job = os.path.join(params['kube']['job_files'], job_name + ".yaml")
    save_file(path_job, filled_template)

    #subprocess.run(['kubectl', 'apply', '-f', path_job])

if __name__=="__main__":

    args = parse_args(sys.argv)

    params = load_config(args['config'])

    run(params)
