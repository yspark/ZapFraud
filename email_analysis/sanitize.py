#!/usr/bin/python
import os
import uuid
from common import *
from bs4 import BeautifulSoup

directory = '../email_data'
count = 0
print 'DIRECTORY: ', directory



for (dirpath, dirnames, filenames) in os.walk(directory):
  for filename in filenames:
    count+=1
    if count % 5000 == 0:
      print "processing... %d" % count
    full_filename = dirpath + '/' + filename
    #print full_filename
    if re.match(r'[0-9]+[.]', filename) == None: continue


    fromAddr = srcIp = toAddr = subject = msgId = date = language = replyToAddr = category = body = None


    phase = 'head'
    for line in open(full_filename).readlines():
      line = line.strip()
      if phase == 'head':
        if line.startswith('Message-ID:'):
          msgId = line.split(":")[1].strip()
        elif line.startswith("Date:"):
          date = line.split(":")[1].strip()
        elif line.startswith("From:"):
          fromAddr = line.split(":")[1].strip()
        elif line.startswith("Subject:"):
          subject = line.split(":")[1].strip()
        elif 'charset' in line:
          language = line.split("=")[-1].strip()
        elif line == '':
          phase = 'body'
          body = ''
        #endif
      elif phase == 'body':
        if line.startswith('-----'): break
        body += line + '\n'
    #endfor


    if body and len(body) > 2:
      # remove HTML tags
      body = BeautifulSoup(body).get_text().encode('utf-8').strip()

      #print dirpath, filename
      if len(dirpath.split('/')) < 6: continue
      output_filename = '../email_data/enron/' + dirpath.split('/')[4] + '_' + dirpath.split('/')[5] + '_' + filename + 'txt'
      uid = str(uuid.uuid1())
      category = 'Enron'
      '''
      print '============================='
      print "FILE: ", output_filename
      print "UUID: ", uid
      print "FROM: ", fromAddr
      print "SRC IP: ", srcIp
      print "TO: ", toAddr
      print "SUBJECT: ", subject
      print "MSGID: ", msgId
      print "DATE: ", date
      print "LANGUAGE: ", language
      print "REPLY-TO: ", replyToAddr
      print "CATEGORY: ", category
      print "BODY: ", body
      print '======================================================================'
      '''
      file_output(output_filename, uid, fromAddr, srcIp, toAddr, subject, msgId, date, language, replyToAddr, category, body)

