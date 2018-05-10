import configparser

import helper.file_helper as file_manager

__author__ = 'garrett_local'


def write_cfg_file(path, cfg):
    """
    Write configuration to a txt file.
    :param path: basestring. Indicating where to save the file.
    :param cfg: configuration. Must be a dict containing a few of dict. Each
                dict contained is content of a section. They contain a few
                items, whose value should be basestring.
    """
    conf = configparser.ConfigParser()
    file_manager.create_dirname_if_not_exist(path)
    cfg_file = open(path, 'w')
    sections = cfg.keys()
    sections = sorted(sections)
    for section in sections:
        content = cfg[section]
        conf.add_section(section)
        items_names = content.keys()
        items_names = sorted(items_names)
        for item_name in items_names:
            conf.set(section, item_name, str(content[item_name]))
    conf.write(cfg_file)
    cfg_file.close()


def write_exp_cfg_file(cfg, exp_name, cfg_name):
    path = './result/tmp/{}/cfg/{}'.format(exp_name, cfg_name)
    write_cfg_file(path, cfg)


def read_cfg_file(path):
    """
    Read configuration saved by function write_cfg_file.
    :param path: basestring. Where to find the configuration file.
    :return: configuration. It is a dict containing a few of dict. Each dict
                contained is content of a section. They contain a few items,
                whose value will be basestring.
    """
    conf = configparser.ConfigParser()
    conf.read(path)
    sections = conf.sections()
    cfg = {}
    if len(sections) == 0:
        raise AssertionError('sections is empty. File {} may not exist or may '
                             'be empty'.format(path))
    for section in sections:
        options = conf.options(section)
        cfg_section = {}
        for option in options:
            item = conf.get(section, option)
            cfg_section[option] = item
        cfg[section] = cfg_section
    return cfg


def to_bool(s):
    if isinstance(s, bool):
        return s
    elif isinstance(s, str):
        return True if s == 'True' else False
    else:
        raise ValueError
