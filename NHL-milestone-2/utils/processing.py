import os 
import pandas as pd 
import numpy as np

def getData():
  """
  Retruns the data from 2015 to 2018 for training and validation, 2019 for testing
  Input: None
  Output: 
  - two arrays of shape (m x n), (m x q)
  """
  df15r = pd.read_csv('./2015R.csv')
  df16r = pd.read_csv('./2016R.csv')
  df17r = pd.read_csv('./2017R.csv')
  df18r = pd.read_csv('./2018R.csv')
  df19r = pd.read_csv('./2019R.csv')

  frames = [df15r, df16r, df17r, df18r]
  df_dataset_R = pd.concat(frames)
  df_dataset_R = df_dataset_R.drop(['Unnamed: 0'], axis=1)
  
  df_test_R = df19r
  df_test_R = df_test_R.drop(['Unnamed: 0'], axis=1)
  return (df_dataset_R,df_test_R)

def get_goal_coordinates(rinkSide, opposite = False): 
    """
    Input: 
    - rinkSide: The column "rinkSide" which is part of the dataframe; indicates which side of the rink the home-team starts (period == 1) 
    - opposite: boolean
    Output: 
    - An array of shape (1 x 2); [x, y];  containing the coordinates of the goal post
    """
    left_goal = np.array([-89, 0])
    right_goal = np.array([89, 0]) 
    if opposite == False: 
        if rinkSide == "left": 
            return right_goal
        else: 
            return left_goal
    else: 
        if rinkSide == "left": 
            return left_goal
        else: 
            return right_goal
def get_shot_distance(df2): 
    """
    Returns the distance of shot from the (target) goal for each shot in the dataframe
    Input: 
    - df2: dataframe; shape (m x n) 
    Output: 
    - np-array; shape (m x 1) 
    """
    distance = []
    for shot in range(len(df2)): 
        shot_coordinates = np.array([df2.loc[shot, 'coordinates x'], df2.loc[shot, 'coordinates y']])
        if (df2.loc[shot, 'period num'].item() % 2) == 1: 
            if df2.loc[shot, 'team name'] == df2.loc[shot, 'home name']:
                distance.append(np.linalg.norm(shot_coordinates - get_goal_coordinates(df2.loc[shot, 'rinkSide'], opposite = False)))
            else: 
                distance.append(np.linalg.norm(shot_coordinates - get_goal_coordinates(df2.loc[shot, 'rinkSide'], opposite = True)))
        else: 
            if df2.loc[shot, 'team name'] == df2.loc[shot, 'home name']:# rinkSide
                distance.append(np.linalg.norm(shot_coordinates - get_goal_coordinates(df2.loc[shot, 'rinkSide'], opposite = True)))
            else: 
                distance.append(np.linalg.norm(shot_coordinates - get_goal_coordinates(df2.loc[shot, 'rinkSide'], opposite = False)))
            
    return distance

def get_direction(shot):
  """
  Return to the side of the net to which the shot was directed
  input:
  - shot: a row of dataframe; shape (m x 1)
  Output:
  - bool: True for the left side, False for the right side
  """
  if (shot['period num'].item() % 2) == 1:
    if (shot['team name'] == shot['home name'] and shot['rinkSide'] == 'right') or (shot['team name'] == shot['away name'] and shot['rinkSide'] == 'left'):
      return True
    else:
      return False
  else:
    if (shot['team name'] == shot['home name'] and shot['rinkSide'] == 'right') or (shot['team name'] == shot['away name'] and shot['rinkSide'] == 'left'):
      return False
    else:
      return True


def get_shot_angle(df2): 
  """
  Returns the angle of shot from the (target) goal for each shot in the dataframe
  left goal net: Clockwise [0,360]
  right goal net: anticlockwise [0,360]
  Input: 
  - df2: dataframe; shape (m x n) 
  Output: 
  - np-array; shape (m x 1) 
  """
  angle = []
  for shot in range(len(df2)): 
      if(df2.loc[shot, 'Distance from net'] == 0):
          angle.append(0)
      else:
          if get_direction(df2.loc[shot]):
              if df2.loc[shot, 'coordinates x'] >= -89:
                  angle.append(np.degrees(np.arccos(df2.loc[shot, 'coordinates y'] / df2.loc[shot, 'Distance from net'])))
              else:
                  angle.append(180 + np.degrees(np.arccos( - df2.loc[shot, 'coordinates y'] / df2.loc[shot, 'Distance from net'])))
          else:
              if df2.loc[shot, 'coordinates x'] <= 89:
                  angle.append(np.degrees(np.arccos( - df2.loc[shot, 'coordinates y'] / df2.loc[shot, 'Distance from net'])))
              else:
                  angle.append(180 + np.degrees(np.arccos(df2.loc[shot, 'coordinates y'] / df2.loc[shot, 'Distance from net'])))
  return angle


def addNewCol(df):
  #Distance from net
  distances = [] 
  for name,group in df_dataset_R.groupby('season'):
    distances.append(get_shot_distance(group))  
  df.insert(20, 'Distance from net', distances[0]+distances[1]+distances[2]+distances[3])

  #Angle from net
  angles = []
  for _,group in df_dataset_R.groupby('season'):
    angles.append(get_shot_angle(group))  
  df.insert(21, 'Angle from net', angles[0]+angles[1]+angles[2]+angles[3])

  #is goal(0 or 1)
  isGoal = []
  shot_or_goal = df['shot or goal'].tolist()
  for i in shot_or_goal:
    if i == 'Goal':
      isGoal.append(1)
    else:
      isGoal.append(0)
  df.insert(22, 'Is goal', isGoal)

  #Empty Net (0 or 1; NaNs are 0)
  isEmptyNet = []
  empty_net = df['empty net'].tolist()
  for i in empty_net:
    if i == 'True':
      isEmptyNet.append(1)
    else:
      isEmptyNet.append(0)
  df.insert(23, 'Empty Net', isEmptyNet)
  df = df.drop(['empty net'],axis = 1)

  return df

df_dataset_R,df_test_R = getData()

addNewCol(df_dataset_R)
df_dataset_R = df_dataset_R.reset_index(drop=True)
df_dataset_R.to_csv('./Milestone2_newData.csv')

addNewCol(df_test_R)
df_test_R = df_test_R.reset_index(drop=True)
df_test_R.to_csv('./Milestone2_newTestData.csv')
