def timeToSec(time, per):
  secs = []
  for i in range(len(time)):
    sec = datetime.datetime.strptime(time[i], '%M:%S').minute*60 + datetime.datetime.strptime(time[i], '%M:%S').second
    if per[i] == 1:
      secs.append(sec)
    elif per[i] == 2:
      secs.append(20*60 + sec)
    elif per[i] == 3:
      secs.append(40*60 + sec)
    elif per[i] == 4:
      secs.append(60*60 + sec)
    elif per[i] == 5:
      secs.append(65*60 + sec)
  return secs

def getData_Q4(): 
  """
  input: None
  output: dataframe with 8 columns
  """
  df_q2 = pd.read_csv('./Milestone2_newData.csv')
  secs = timeToSec(df_q2['period time'].tolist(), df_q2['period num'].tolist())
  df = df_q2[['period num', 'coordinates x', 'coordinates y', 'Distance from net', 'Angle from net', 'shot type']]
  df.insert(0, 'period time', secs)
  return df

import json
def jsonToDataframe(fileName):
  '''
  input: str -> json fileName
  output: dataframe shape(m,n) depends on json file
  Convert the JSON file to a Dataframe
  '''
  with open(fileName,'r') as f:
    data = json.loads(f.read())
  df= pd.json_normalize(data, ['liveData', 'plays', 'allPlays'], ['gamePk'], errors='ignore')
  try:
    df = df[['about.periodTime', 'about.period', 'about.eventId', 'gamePk', 'team.name', 'result.event', 'coordinates.x', 'coordinates.y', 'result.secondaryType']]
    #df = df[(df['result.event'] == 'Shot')|(df['result.event'] == 'Goal')]
    df.insert(3, 'season', data['gameData']['game']['season'])
    df.insert(6, 'home name', data['gameData']['teams']['home']['name'])
    df.insert(7, 'away name', data['gameData']['teams']['away']['name'])
    df.insert(8, 'rinkSide', data['liveData']['linescore']['periods']['num' == 1]['home']['rinkSide'])
  except(KeyError):
    df = pd.DataFrame(columns=['about.periodTime', 'about.period', 'about.eventId', 'gamePk', 'team.name', 'result.event', 'coordinates.x', 'coordinates.y', 'result.secondaryType'])
  df = df.rename({'about.periodTime': 'period time', 'about.period': 'period num', 'about.eventId': 'event Id', 'gamePk': 'game ID', 'team.name': 'team name', 'result.event': 'shot or goal', 'coordinates.x': 'coordinates x', 'coordinates.y': 'coordinates y',  'result.secondaryType': 'shot type'},
                      axis='columns')

  return df

def getPreviouEvent(df):
  '''
  input: dataframe (with all event)
  output: dataframe (with shot and goal event, and add 4 new features about previous events)
  '''
  df.insert(13, 'last event type', 'NaN')
  df.insert(14, 'last coordinates x', 'NaN')
  df.insert(15, 'last coordinates y', 'NaN')
  df.insert(16, 'Time from the last event', 'NaN')
  df.insert(17, 'last event team name', 'NaN')
  for i in range(1,df.shape[0]):
    if df.loc[i, ['shot or goal']].tolist()[0] == 'Shot' or 'Goal':
      df.loc[i, ['last event type', 'last coordinates x', 'last coordinates y', 'Time from the last event', 'last event team name']] = df.loc[i-1, ['shot or goal','coordinates x', 'coordinates y', 'period time', 'team name']].tolist()[:5]
  df = df[(df['shot or goal'] == 'Shot')|(df['shot or goal'] == 'Goal')]
  df = df.reset_index(drop=True) 
  return df

def eachSeason(start, end):
  '''
  input: int, int 
  output: dataframe each season each year
  '''
  path = "/content/drive/MyDrive/IFT6758/ift6758-project-template-main/data"
  if not os.path.exists("/content/drive/MyDrive/IFT6758/ift6758-project-template-main/data"):
    os.mkdir(path)
  os.chdir(path)

  files= os.listdir(path)
  files.sort()

  df_all = pd.DataFrame(columns=['period time', 'period num', 'event Id', 'season', 'game ID', 'team name', 'home name', 'away name', 'rinkSide', 'shot or goal', 'coordinates x', 'coordinates y', 'shot type'])

  for file in files[start:end]: 
    df_all = pd.concat([df_all, jsonToDataframe(file)])
  df_all = df_all.reset_index(drop = True) 
  df_all = getPreviouEvent(df_all)
  return df_all

def getMS2FE2data():
  '''
  input: None
  Output: None
  download all dataframes save as csv file
  '''
  data = eachSeason(0, 1230)
  data.to_csv('./ms2_2015R.csv')
  data = eachSeason(1325, 2555)
  data.to_csv('./ms2_2016R.csv')
  data = eachSeason(2657, 3887)
  data.to_csv('./ms2_2017R.csv')
  data = eachSeason(3992, 5263)
  data.to_csv('./ms2_2018R.csv')
  data = eachSeason(5368, 6639)
  data.to_csv('./ms2_2019R.csv')

  return

def getms2Data():
  """
  Retruns the data from 2015 to 2018 Regular for training and validation, 2019 for testing
  Input: None
  Output: 
  - two arrays of shape (m x n), (m x q)
  """
  df15r = pd.read_csv('./data/ms2_2015R_1.csv')
  df16r = pd.read_csv('./data/ms2_2016R_1.csv')
  df17r = pd.read_csv('./data/ms2_2017R_1.csv')
  df18r = pd.read_csv('./data/ms2_2018R_1.csv')
  df19r = pd.read_csv('./ms2_2019R_1.csv')

  frames = [df15r, df16r, df17r, df18r]
  df_dataset_ms2 = pd.concat(frames)
  df_dataset_ms2 = df_dataset_ms2.reset_index(drop=True)
  df_dataset_ms2 = df_dataset_ms2.drop(['Unnamed: 0'], axis=1)
  
  df_test_ms2 = df19r
  df_test_ms2 = df_test_ms2.drop(['Unnamed: 0'], axis=1)
  return (df_dataset_ms2,df_test_ms2)

def getLastDistance(df):
  dist = []
  for shot in range(len(df)):
    last = np.array([df.loc[shot, 'last coordinates x'], df.loc[shot, 'last coordinates y']])
    current = np.array([df.loc[shot, 'coordinates x'],df.loc[shot, 'coordinates y']])
    distance = np.linalg.norm(last-current)  
    dist.append(distance)
  df.insert(17, 'Distance from the last event', dist)
  return df

import math
def getAngle(a, b, c):
  angle = []
  for i in range(len(a)):
    if (a[i]+b[i] <= c[i])or(a[i]+c[i] <= b[i])or(b[i]+c[i] <= a[i]):
      angle.append(0)
    else:
      angle.append(math.degrees(math.acos((a[i]*a[i]-b[i]*b[i]-c[i]*c[i])/(-2*b[i]*c[i]))))
  return angle

import datetime
def findspeed(dist, cur, last):
  if(cur - last) == 0:
    return 'nan' # means infinity
  return dist / (cur - last)

def addNewColQ4(df):
  df = df.dropna(axis=0,subset = ['coordinates x', "coordinates y"])
  df.insert(12, 'Distance from net', df_dataset_R['Distance from net'].tolist())
  df.insert(13, 'Angle from net', df_dataset_R['Angle from net'].tolist())
  df.insert(14, 'Is goal', df_dataset_R['Is goal'].tolist())
  #Rebound
  rebound = []
  lastevent = df['last event type'].tolist()
  team = df['team name'].tolist()
  lastteam = df['last event team name'].tolist()
  for i in range(len(lastevent)):
    if lastevent[i] == 'Shot' and team[i] == lastteam[i]:
      rebound.append(True)
    else:
      rebound.append(False)
  df.insert(21, 'Rebound', rebound)

  #get last distance from net
  df_rebound = df[df['Rebound'] == True]
  df_rebound = df_rebound.drop(['coordinates x', 'coordinates y'], axis=1)
  df_rebound = df_rebound.rename({'last coordinates x': 'coordinates x', 'last coordinates y': 'coordinates y'}, axis='columns')
  df_rebound = df_rebound.reset_index()
  distances = [] 
  for _,group in df_rebound.groupby('season'):
    distances.append(get_shot_distance(group))  
  df_rebound.insert(22, 'last Distance from net', distances[0]+distances[1]+distances[2]+distances[3])

  #Change in shot angle
  #df_rebound = df_rebound.dropna(axis=0, subset = ['coordinates x', "coordinates y"])
  a = df_rebound['Distance from the last event'].tolist()
  b = df_rebound['Distance from net'].tolist()
  c = df_rebound['last Distance from net'].tolist()
  angles = getAngle(a,b,c)
  #df = df.dropna(axis=0, subset = ['coordinates x', "coordinates y", 'last coordinates x', "last coordinates y"])
  df.insert(21, 'Change in shot angle', 0)
  df = df.reset_index(drop=True)
  index = 0
  for i in range(len(df)):
    if df.loc[i, 'Rebound'] == True:
      if angles[index] != 0:
        df.loc[i, 'Change in shot angle'] = angles[index]
      else:
        df.loc[i, 'Rebound'] == False
      index = index + 1
  
  #speed
  speed = []
  dist = df['Distance from the last event'].tolist()
  per = df['period num'].tolist()
  cur = timeToSec(df['period time'].tolist(), per) 
  last = timeToSec(df['Time from the last event'].tolist(), per)

  for i in range(len(dist)):
    s = findspeed(dist[i], cur[i], last[i])
    speed.append(s)
  df.insert(23, 'Speed', speed)
  return df

df_dataset_R = df_dataset_R.dropna(axis=0,subset = ['coordinates x', "coordinates y"])
df_dataset_ms2 = addNewColQ4(df_dataset_ms2)
df_final = df_dataset_ms2[['period time', 'period num', 'coordinates x', 'coordinates y', 'Distance from net', 'Angle from net', 'shot type', 'last event type', 'last coordinates x', 'last coordinates y', 'Time from the last event', 'Distance from the last event', 'Rebound', 'Change in shot angle', 'Speed', 'Is goal']]
#df_final.to_csv('../ms2Q4.csv')
df_final = df_final.dropna(axis=0,subset = ['shot type', 'last coordinates x', 'last coordinates y', 'Speed'])
df_final = df_final.reset_index(drop=True)
df_final
df_final.to_csv('../ms2Q4.csv')

import comet_ml
from comet_ml import Experiment


subset_df = pd.read_csv('./wpg_v_wsh_2017021065.csv')
experiment = comet_ml.Experiment("FkBHlm1Ewg6VtxJZ4nXYxN0U1")
experiment = Experiment(  
    api_key=('FkBHlm1Ewg6VtxJZ4nXYxN0U1'),
    project_name='milestone-2',
    workspace= 'maskedviper',
)
experiment.log_dataframe_profile(
    subset_df, 
    name='wpg_v_wsh_2017021065',  # keep this name
    dataframe_format='csv'  # ensure you set this flag!
)
