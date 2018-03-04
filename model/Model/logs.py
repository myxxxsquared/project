import logging
import os
from functools import partial


def print_(content, func):
    print(content)
    func(content)


def logging_init(configs):
    path = os.path.join(configs.log_path, configs.code_name + '_running.log')
    if os.path.isfile(path):
        os.system('rm %s' % path)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=path,
                        filemode='w')
    for k, v in configs.__dict__.items():
        logging.info('%s: %s' % (k, str(v)))
    if configs.print_on_screen:
        return {'info': partial(print_, func=logging.info), 'debug': partial(print_, func=logging.debug)}
    else:
        return {'info': logging.info, 'debug': logging.debug}
