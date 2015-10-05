#!/usr/bin/python

import os, sys, re, csv
import numpy as np
from dateutil import parser

TOTAL_KEYWORDS = "_TOTAL_KEYWORDS_"

'''
DIR_LIST = ["../../email_data/419baiter/", "../../email_data/antifraud/", "../../email_data/scamdex/", "../../email_data/scammedby/", "../../email_data/scamwarners/"]
DIR_LIST += ["../../email_data_year_month/year_month/enron/", "../../email_data_year_month/year_month/jebbush/", "../../email_data_year_month/year_month/amazon_review/"]
'''

DIR_LIST = ["../../email_data/antifraud/"]

YEAR_LIST = np.arange(2000, 2016)
MONTH_LIST = np.arange(1, 13)

word_list = []
for line in open("words.txt").readlines():
	word_list.append(line.strip().lower())

# word_count_dic[Source][Category][Year][Month][Word]
word_count_dic = {}
# email_count_dic[Source][Category][Year][Month][Word]
email_count_dic = {}
# total_word_count_dic[Source][Category][Year][Month]
total_word_count_dic = {}
# total_word_count_dic[Source][Category][Year][Month]
total_email_count_dic = {}
# word_ratio_dic[Source][Category][Year][Month][Word]
word_ratio_dic = {}
# email_ratio_dic[Source][Category][Year][Month][Word]
email_ratio_dic = {}


for directory in DIR_LIST:
	source = directory.split("/")[-2]

	word_count_dic[source] = {}
	email_count_dic[source] = {}
	total_word_count_dic[source] = {}
	total_email_count_dic[source] = {}
	word_ratio_dic[source] = {}
	email_ratio_dic[source] = {}

	for (dirpath, dirnames, filenames) in os.walk(directory):
		for filename in filenames:
			email_counted = False
			phase = "look_for_date"
			category = ""
			year = ""
			month = ""
			category = ""
			body = ""
			for line in open(dirpath + "/" + filename).readlines():
				line = line.strip().lower()
				if phase == "look_for_date" and line.startswith("date:"):
					line = line.replace("date:", "")
					line = line.replace("gmt", "")
					line = re.sub("\(.*\)", "", line)
					line = re.sub("\(.*$", "", line)
					line = re.sub("\D*$", "", line)
					line = line.replace(".", ":")

					#scamdex
					if "scamdex" in source:
						line = re.sub(" \d$", "", line)
						line = re.sub(" \d\d$", "", line)
					#endif

					#scamwarners
					if "scamwarners" in source:
						line = re.sub("at \d+$", "", line)
						line = re.sub(" \d$", "", line)
						line = re.sub(" \d\d$", "", line)
					#endif

					line = line.strip()
					if len(line) < 5: break
					try:
						date_obj = parser.parse(line)
						year = date_obj.year
						month = date_obj.month
						#print "VALID DATE: ", line, year, month
					except:
						print dirpath + "/" + filename
						print "INVALID DATE:--" + line + "--"
						continue
					phase = "look_for_category"
				elif phase == "look_for_category" and line.startswith("category:"):
					category = line.replace("category:", "").strip()
					category = category.replace("/", "_")
					if len(category) < 2: category = "Unknown"
					phase = "look_for_body"
				elif phase == "look_for_body" and line.startswith("body:"):
					body += line.replace("body:", "").strip().lower()
					body += " "
					phase = "body"
				elif phase == "body":
					body += line.strip().lower()
					body += " "
			#end for

			if year not in YEAR_LIST: continue
			if month not in MONTH_LIST: continue
			if phase != "body": continue

			body = body.replace("  ", " ")

			# initialize word_count_dic for specific category
			if category not in email_count_dic[source].keys():
				word_count_dic[source][category] = {}
				email_count_dic[source][category] = {}				
				total_word_count_dic[source][category] = {}
				total_email_count_dic[source][category] = {}
				word_ratio_dic[source][category] = {}
				email_ratio_dic[source][category] = {}

				for y in YEAR_LIST:
					word_count_dic[source][category][y] = {}
					email_count_dic[source][category][y] = {}
					total_word_count_dic[source][category][y] = {}
					total_email_count_dic[source][category][y] = {}
					word_ratio_dic[source][category][y] = {}
					email_ratio_dic[source][category][y] = {}

					for m in MONTH_LIST:
						word_count_dic[source][category][y][m] = {}
						email_count_dic[source][category][y][m] = {}						
						total_word_count_dic[source][category][y][m] = 0
						total_email_count_dic[source][category][y][m] = 0
						word_ratio_dic[source][category][y][m] = {}
						email_ratio_dic[source][category][y][m] = {}

						for word in word_list:
							word_count_dic[source][category][y][m][word] = 0
							email_count_dic[source][category][y][m][word] = 0
							word_ratio_dic[source][category][y][m][word] = 0
							email_ratio_dic[source][category][y][m][word] = 0
						#end for word
						word_count_dic[source][category][y][m][TOTAL_KEYWORDS] = 0
						email_count_dic[source][category][y][m][TOTAL_KEYWORDS] = 0
			#endif

			# count valid words
			for word in word_list:
				if word in body:
					word_count_dic[source][category][year][month][word] += 1
					word_count_dic[source][category][year][month][TOTAL_KEYWORDS] += 1
					if email_counted == False:
						email_count_dic[source][category][year][month][word] += 1
						email_count_dic[source][category][year][month][TOTAL_KEYWORDS] += 1
						email_counted = True

			# total words
			total_word_count_dic[source][category][year][month] = len(body.split())
			# total emails
			total_email_count_dic[source][category][year][month] += 1
		#end for filename
	#end for
#end for

# word,email ratio
for s in word_ratio_dic.keys():
	for c in word_ratio_dic[s].keys():
		for y in word_ratio_dic[s][c]:
			for m in word_ratio_dic[s][c][y]:
				for w in word_ratio_dic[s][c][y][m]:
					if total_word_count_dic[s][c][y][m] != 0:
						word_ratio_dic[s][c][y][m][w] = float(word_count_dic[s][c][y][m][w]) / float(total_word_count_dic[s][c][y][m])
						email_ratio_dic[s][c][y][m][w] = float(email_count_dic[s][c][y][m][w]) / float(total_email_count_dic[s][c][y][m])

'''
# word,email ratio to csv (year_month)
header = ["word/date"]
for y in YEAR_LIST:
	for m in MONTH_LIST:
		header.append(str(y)+"_"+str(m))

for s in word_ratio_dic.keys():
	for c in word_ratio_dic[s].keys():
		csvfile = open("./csv/"+s+"_"+c+".csv", "wb")
		csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
		csvwriter.writerow(header)

		for w in word_list:
			row = []
			row.append(w)
			for y in YEAR_LIST:
				for m in MONTH_LIST:
					row.append(word_ratio_dic[s][c][y][m][w])
			csvwriter.writerow(row)
		#endfor
	#endfor
#endfor
'''

					
				


