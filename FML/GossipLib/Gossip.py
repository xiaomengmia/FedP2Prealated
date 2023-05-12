# -*- coding: utf-8 -*-
import sys, os
import datetime, time

class Gossip(object):

  def __init__(self):
    super(Gossip, self).__init__()
    return

# def RandomPeerSelection(self, Hosts, Exclude):
#   host = ''
#    TmpHosts = []
#    for host in Hosts:
#      if host in Exclude:
#        TmpHosts.append(host)
#    host = random.choices(TmpHosts, k=1)
#    return host[0]
