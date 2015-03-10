NUM_LABLE = 9
MAX_EMAIL_PER_DIR = 1000

TFIDF_DIR = './tfidf'

CATEGORY_DIRECTORY_LIST = ['../email_data/scamdex', '../email_data/stopscammers', '../email_data/malescammers', '../email_data/antifraud', '../email_data/enron']
#CATEGORY_DIRECTORY_LIST = ['../email_data/antifraud']


def get_label(category):
  if any([category == 'Advance Fee Fraud/419', category == 'AdvanceFeeFraud']): return 0    # 0
  elif any([category == 'Lotto/Lottery', category == 'Lottery']): return 1
  elif any([category == 'Employment/Job', category == 'Employment']): return 2
  elif any([category == 'Romance']): return 3
  elif any([category == 'StrandedTraveler']): return 4
  elif any([category == 'Phishing, ID Theft']): return 5
  elif any([category == 'Enron' ]): return 6
  elif any([category == 'Rental']): return 7
  elif any([category == 'AuctionSales']): return 8

  else: return -1
