import requests
import time
import pandas as pd
import json
import numpy as np
import datetime
import math
import ift6758.client.serving_client as sc



class GameClient:
    
    def __init__(self, ip: str = "127.0.0.1", port: int = 5000):
        self.ip = ip
        self.port = port
        self.client = sc.ServingClient(self.ip, self.port)

    def get_full_event(self, gameID):
      '''
      input: int 
      output: dataframe shape(m,n) depends on json file
      Get gameID data, save as JSON, convert the JSON file to a Dataframe
      '''
      http = 'https://statsapi.web.nhl.com/api/v1/game/' + str(gameID) + '/feed/live'
      r = requests.get(http)
      if (r.status_code == 200):
        fileName = str(gameID) + '.json'
      try:
        with open(fileName,'wb') as f:
          f.write(r.content)
          with open(fileName,'r') as f:
            data = json.loads(f.read())
          df= pd.json_normalize(data, ['liveData', 'plays', 'allPlays'], ['gamePk'], errors='ignore')
          try:
            df = df[['about.periodTime', 'about.period', 'about.eventId', 'gamePk', 'team.name', 'result.event', 'coordinates.x', 'coordinates.y', 'result.secondaryType','result.emptyNet']]
            #df = df[(df['result.event'] == 'Shot')|(df['result.event'] == 'Goal')]
            df.insert(3, 'season', data['gameData']['game']['season'])
            df.insert(6, 'home name', data['gameData']['teams']['home']['name'])
            df.insert(7, 'away name', data['gameData']['teams']['away']['name'])
            df.insert(8, 'rinkSide', data['liveData']['linescore']['periods']['num' == 1]['home']['rinkSide'])
          except(KeyError):
            df = pd.DataFrame(columns=['about.periodTime', 'about.period', 'about.eventId', 'gamePk', 'team.name', 'result.event', 'coordinates.x', 'coordinates.y', 'result.secondaryType', 'result.emptyNet'])
          df = df.rename({'about.periodTime': 'period time', 'about.period': 'period num', 'about.eventId': 'event Id', 'gamePk': 'game ID', 'team.name': 'team name', 'result.event': 'shot or goal', 'coordinates.x': 'coordinates x', 'coordinates.y': 'coordinates y',  'result.secondaryType': 'shot type', 'result.emptyNet': 'empty net'},
                              axis='columns')
          return df
      except Exception as e:
        print("Wrong GameID")



    def shot_with_pre_event(self, gameID):
      '''
      input: dataframe (with all event)
      output: dataframe (with shot and goal event, and add 4 new features about previous events)
      '''
      df = self.get_full_event(gameID)

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

    def getLastDistance(self, df):
      dist = []
      for shot in range(len(df)):
        last = np.array([df.loc[shot, 'last coordinates x'], df.loc[shot, 'last coordinates y']])
        current = np.array([df.loc[shot, 'coordinates x'],df.loc[shot, 'coordinates y']])
        distance = np.linalg.norm(last-current)  
        dist.append(distance)
      df.insert(18, 'Distance from the last event', dist)
      return df
      
    def timeToSec(self, time, per):
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

    
    def getAngle(self, a, b, c):
      angle = []
      for i in range(len(a)):
        if (a[i]+b[i] <= c[i])or(a[i]+c[i] <= b[i])or(b[i]+c[i] <= a[i]):
          angle.append(0)
        else:
          angle.append(math.degrees(math.acos((a[i]*a[i]-b[i]*b[i]-c[i]*c[i])/(-2*b[i]*c[i]))))
      return angle

 
    def findspeed(self, dist, cur, last):
      if(cur - last) == 0:
        return 0 # means infinity
      return dist / (cur - last)

    def get_goal_coordinates(self, rinkSide, opposite = False): 
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

    def get_shot_distance(self, df2): 
        """
        Returns the distance of shot from the (target) goal for each shot in the dataframe
        Input: 
        - df2: dataframe; shape (m x n) 
        Output: 
        - np-array; shape (m x 1) 
        """
        df2 = df2.reset_index(drop=True)
        distance = []
        for shot in range(len(df2)): 
            shot_coordinates = np.array([df2.loc[shot, 'coordinates x'], df2.loc[shot, 'coordinates y']])
            if (df2.loc[shot, 'period num'].item() % 2) == 1: 
                if df2.loc[shot, 'team name'] == df2.loc[shot, 'home name']:
                    distance.append(np.linalg.norm(shot_coordinates - self.get_goal_coordinates(df2.loc[shot, 'rinkSide'], opposite = False)))
                else: 
                    distance.append(np.linalg.norm(shot_coordinates - self.get_goal_coordinates(df2.loc[shot, 'rinkSide'], opposite = True)))
            else: 
                if df2.loc[shot, 'team name'] == df2.loc[shot, 'home name']:# rinkSide
                    distance.append(np.linalg.norm(shot_coordinates - self.get_goal_coordinates(df2.loc[shot, 'rinkSide'], opposite = True)))
                else: 
                    distance.append(np.linalg.norm(shot_coordinates - self.get_goal_coordinates(df2.loc[shot, 'rinkSide'], opposite = False)))
                
        return distance

    def get_direction(self, shot):
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

    def get_shot_angle(self, df2): 
      """
      Returns the angle of shot from the (target) goal for each shot in the dataframe
      left goal net: Clockwise [0,360]
      right goal net: anticlockwise [0,360]
      Input: 
      - df2: dataframe; shape (m x n) 
      Output: 
      - np-array; shape (m x 1) 
      """
      df2 = df2.reset_index(drop=True)
      angle = []
      for shot in range(len(df2)): 
          if(df2.loc[shot, 'Distance from net'] == 0):
              angle.append(0)
          else:
              if self.get_direction(df2.loc[shot]):
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

    def addNewCol(self, df):
      '''
      Adding four new features: Distance from net, Angle from net, is goal, Empty Net
      Input: an array of shape (m x n)
      Output: an array of shape ((m+4) x n)
      '''
      #Distance from net
      distances = [] 
      for name,group in df.groupby('season'):
        distances.append(self.get_shot_distance(group))  
      df.insert(18, 'Distance from net', distances[0])

      #Angle from net
      angles = []
      for _,group in df.groupby('season'):
        angles.append(self.get_shot_angle(group))  
      df.insert(19, 'Angle from net', angles[0])

      #is goal(0 or 1)
      isGoal = []
      shot_or_goal = df['shot or goal'].tolist()
      for i in shot_or_goal:
        if i == 'Goal':
          isGoal.append(1)
        else:
          isGoal.append(0)
      df.insert(20, 'Is goal', isGoal)

      #Empty Net (0 or 1; NaNs are 0)
      isEmptyNet = []
      empty_net = df['empty net'].tolist()
      for i in empty_net:
        if i == 'True':
          isEmptyNet.append(1)
        else:
          isEmptyNet.append(0)
      df.insert(21, 'Empty Net', isEmptyNet)
      df = df.drop(['empty net'],axis = 1)
      return df


    def addNewColQ4(self, df):
      df = self.addNewCol(df)
      df = self.getLastDistance(df)
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
      df.insert(18, 'Rebound', rebound)
      df = df.dropna(axis=0, subset = ['coordinates x', "coordinates y", 'last coordinates x', "last coordinates y", 'shot type'])

      #get last distance from net
      df_rebound = df[df['Rebound'] == True]
      df_rebound = df_rebound.drop(['coordinates x', 'coordinates y'], axis=1)
      df_rebound = df_rebound.rename({'last coordinates x': 'coordinates x', 'last coordinates y': 'coordinates y'}, axis='columns')
      df_rebound = df_rebound.reset_index()
      distances = [] 
      for _,group in df_rebound.groupby('season'):
        distances.append(self.get_shot_distance(group))  
      df_rebound.insert(19, 'last Distance from net', distances[0])#distances[0]+distances[1]+distances[2]+distances[3])

      #Change in shot angle
      #df_rebound = df_rebound.dropna(axis=0, subset = ['coordinates x', "coordinates y"])
      a = df_rebound['Distance from the last event'].tolist()
      b = df_rebound['Distance from net'].tolist()
      c = df_rebound['last Distance from net'].tolist()
      angles = self.getAngle(a,b,c)
      #df = df.dropna(axis=0, subset = ['coordinates x', "coordinates y", 'last coordinates x', "last coordinates y"])
      df.insert(20, 'Change in shot angle', 0)
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
      cur = self.timeToSec(df['period time'].tolist(), per) 
      last = self.timeToSec(df['Time from the last event'].tolist(), per)
      for i in range(len(dist)):
        s = self.findspeed(dist[i], cur[i], last[i])
        speed.append(s)
      df.insert(21, 'Speed', speed)
      return df

    def final_df(self, df):
      df = df[['home name','away name','team name','period time', 'period num', 'coordinates x', 'coordinates y', 'Distance from net', 'Angle from net', 'shot type', 'last event type', 'last coordinates x', 'last coordinates y', 'Time from the last event', 'Distance from the last event', 'Rebound', 'Change in shot angle', 'Speed', 'Is goal']]
      df = df.dropna(axis=0,subset = ['shot type', 'last coordinates x', 'last coordinates y', 'Speed'])
      df = df.reset_index(drop=True)

      pt = df.pop('period time').tolist()
      pn = df['period num'].tolist()
      secs = self.timeToSec(pt,pn)
      df.insert(3, 'period time', secs)
      lastt = df.pop('Time from the last event').tolist()
      lastsecs = self.timeToSec(lastt,pn)
      diff = []
      for i in range(len(secs)):
        diff.append(secs[i]-lastsecs[i])
      df.insert(10, 'Time from the last event', diff)
      return df

    def final_df_one_hot(self, df):
      df = self.final_df(df)
      df = df.dropna(axis=0, how='any')
      df_one_hot = df[['shot type','last event type','Rebound']]
      shotType = pd.get_dummies(df['shot type'],prefix='shotType')
      lastEventType = pd.get_dummies(df['last event type'],prefix='lastEventType')
      Rebound = pd.get_dummies(df['Rebound'],prefix='Rebound')
      df = df.drop(['shot type','last event type','Rebound'], axis=1)
      df_one_hot = pd.concat([df, shotType, lastEventType, Rebound], axis=1)
      isGoal = df_one_hot.pop('Is goal')
      if 'shotType_Backhand' not in list(df_one_hot):
        df_one_hot.insert(15, 'shotType_Backhand', 0)
      if 'shotType_Deflected' not in list(df_one_hot):
        df_one_hot.insert(16, 'shotType_Deflected', 0)
      if 'shotType_Slap Shot' not in list(df_one_hot):
        df_one_hot.insert(17, 'shotType_Slap Shot', 0)
      if 'shotType_Snap Shot' not in list(df_one_hot):
        df_one_hot.insert(18, 'shotType_Snap Shot', 0)
      if 'shotType_Tip-In' not in list(df_one_hot):
        df_one_hot.insert(19, 'shotType_Tip-In', 0)
      if 'shotType_Wrap-around' not in list(df_one_hot):
        df_one_hot.insert(20, 'shotType_Wrap-around', 0)
      if 'shotType_Wrist Shot' not in list(df_one_hot):
        df_one_hot.insert(21, 'shotType_Wrist Shot', 0)
      if 'lastEventType_Blocked Shot' not in list(df_one_hot):
        df_one_hot.insert(22, 'lastEventType_Blocked Shot', 0)
      if 'lastEventType_Faceoff' not in list(df_one_hot):
        df_one_hot.insert(23, 'lastEventType_Faceoff', 0)
      if 'lastEventType_Giveaway' not in list(df_one_hot):
        df_one_hot.insert(24, 'lastEventType_Giveaway', 0)
      if 'lastEventType_Goal' not in list(df_one_hot):
        df_one_hot.insert(25, 'lastEventType_Goal', 0)
      if 'lastEventType_Hit' not in list(df_one_hot):
        df_one_hot.insert(26, 'lastEventType_Hit', 0)
      if 'lastEventType_Missed Shot' not in list(df_one_hot):
        df_one_hot.insert(27, 'lastEventType_Missed Shot', 0)
      if 'lastEventType_Penalty' not in list(df_one_hot):
        df_one_hot.insert(28, 'lastEventType_Penalty', 0)
      if 'lastEventType_Shot' not in list(df_one_hot):
        df_one_hot.insert(29, 'lastEventType_Shot', 0)
      if 'lastEventType_Takeaway' not in list(df_one_hot):
        df_one_hot.insert(30, 'lastEventType_Takeaway', 0)
      if 'Rebound_False' not in list(df_one_hot):
        df_one_hot.insert(31, 'Rebound_False', 0)
      if 'Rebound_True' not in list(df_one_hot):
        df_one_hot.insert(32, 'Rebound_False', 0)
      df_one_hot.insert(33, 'Is goal', isGoal)
      return df_one_hot

    def final_df_normalize(self, df):
      df = self.final_df_one_hot(df)
      df_st = df[['period time','period num','coordinates x','coordinates y','Distance from net','Angle from net','last coordinates x','last coordinates y','Time from the last event','Distance from the last event','Change in shot angle','Speed']]  
      df_st = (df_st - df_st.mean()) / (df_st.std())
      df_else = df.drop(['period time','period num','coordinates x','coordinates y','Distance from net','Angle from net','last coordinates x','last coordinates y','Time from the last event','Distance from the last event','Change in shot angle','Speed'], axis=1)
      df_st = pd.concat([df_st,df_else], axis=1)
      return df

    pd.set_option('mode.chained_assignment', None)
    def ping_game(self, game_id, idx, other):
      '''
      input  -> game_id: int    
                idx: int (index of this event)     
                other: dataframe (the past events)
      output -> df_this: series (this event series)    
                new_idx: int (next event index)     
                other: dataframe (the past events + this event)
      '''

      df = self.shot_with_pre_event(game_id)
      df = self.addNewColQ4(df)
      df_empty = pd.DataFrame(columns=['gameID','home name','away name','team name','period time','period num','coordinates x','coordinates y','Distance from net','Angle from net','shot type','Time from the last event','last event type','last coordinates x','last coordinates y','Distance from the last event','Rebound','Change in shot angle','Speed','Is goal','Goal probability','home Goal','away Goal'])
      if df.shape[0] == other[other['gameID'] == game_id].shape[0]:
        print("No new events.")
        return df_empty, df.shape[0], other
      events_xgb = self.final_df_normalize(df).iloc[idx:,3:] 
      events_xgb = events_xgb.reset_index(drop=True)
      df_these= self.final_df(df).iloc[idx:,:]
      df_these = df_these.reset_index(drop=True)
      total_num = df_these.shape[0]
      #max_events_num = 5 #-------------------------
      #if total_num > 5:
        #total_num = 5 #-------------------------
      for i in range(total_num):
        gameID_other = other[other['gameID'] == str(game_id)]
        df_this = df_these.loc[i]
        event_xgb = events_xgb.loc[i]
        df_this = self.add_prob(df_this, event_xgb, gameID_other) 
        other.loc[other.shape[0]] = df_this
        other.loc[other.shape[0]-1,'gameID'] = str(game_id)
      new_idx = df.shape[0]
      return df_this, new_idx, other

    def get_num_and_left_time(self, event): 
      '''
      input: serie - a event
      output: (int, str)
      '''
      if event['period num'] < 5:
        period_secs = event['period time'] - (event['period num']-1) * 20 * 60
        if event['period num'] == 4:
          left_period_secs = 5 * 60 - period_secs
        else:
          left_period_secs = 20 * 60 - period_secs
      else:
        period_secs = event['period time'] - 65 * 60
        left_period_secs = 0
      m, s = divmod(left_period_secs, 60)
      return "%02d:%02d" % (m, s), event['period num']

    def get_team_name(self, event): 
      '''
      input: serie - a event
      output: (str, str)
      '''
      return event['home name'],event['away name']

    def each_goal_prob(self, event): 
      '''
      input: serie - a event
      output: (str, str)
      '''
      return event['home Goal'],event['away Goal']

    def add_prob(self, event, xgb_event, past_events): 
      df_xgb = pd.DataFrame(columns=['period time','period num','coordinates x','coordinates y','Distance from net','Angle from net', 'last coordinates x','last coordinates y',
                                     'Time from the last event', 'Distance from the last event','Change in shot angle','Speed', 
                                     'shotType_Backhand','shotType_Deflected','shotType_Slap Shot','shotType_Snap Shot','shotType_Tip-In','shotType_Wrap-around','shotType_Wrist Shot',
                                     'lastEventType_Blocked Shot','lastEventType_Faceoff','lastEventType_Giveaway','lastEventType_Goal','lastEventType_Hit',
                                     'lastEventType_Missed Shot','lastEventType_Penalty','lastEventType_Shot','lastEventType_Takeaway','Rebound_False','Rebound_True'])
      df_xgb.loc[0] = xgb_event[:-1]
      df_xgb[['period time','period num','Time from the last event']]= df_xgb[['period time','period num','Time from the last event']].astype('int64')
      df_xgb[['shotType_Backhand','shotType_Deflected','shotType_Slap Shot','shotType_Snap Shot','shotType_Tip-In','shotType_Wrap-around','shotType_Wrist Shot',
              'lastEventType_Blocked Shot','lastEventType_Faceoff','lastEventType_Giveaway','lastEventType_Goal','lastEventType_Hit',
              'lastEventType_Missed Shot','lastEventType_Penalty','lastEventType_Shot','lastEventType_Takeaway','Rebound_False','Rebound_True']] = df_xgb[['shotType_Backhand','shotType_Deflected','shotType_Slap Shot','shotType_Snap Shot','shotType_Tip-In','shotType_Wrap-around','shotType_Wrist Shot',
              'lastEventType_Blocked Shot','lastEventType_Faceoff','lastEventType_Giveaway','lastEventType_Goal','lastEventType_Hit',
              'lastEventType_Missed Shot','lastEventType_Penalty','lastEventType_Shot','lastEventType_Takeaway','Rebound_False','Rebound_True']].astype('int64')
      
      df_xgb[['coordinates x','coordinates y','Distance from net','Angle from net','last coordinates x','last coordinates y','Distance from the last event','Change in shot angle','Speed']] = df_xgb[['coordinates x','coordinates y','Distance from net','Angle from net','last coordinates x','last coordinates y','Distance from the last event','Change in shot angle','Speed']].astype('float64')

      prob = float('%.4f' % self.client.predict(df_xgb)['0']['goal confidence'])

      event['Goal probability'] = prob 

      if past_events.shape[0] == 0:
        if event['team name'] == event['home name']:
          event['home Goal'] = prob
          event['away Goal'] = 0
        else:
          event['home Goal'] = 0
          event['away Goal'] = prob
      else:
        if event['team name'] == event['home name']:
          event['home Goal'] = past_events.iloc[past_events.shape[0]-1,21] + prob
          event['away Goal'] = past_events.iloc[past_events.shape[0]-1,22]
        else:
          event['home Goal'] = past_events.iloc[past_events.shape[0]-1,21]
          event['away Goal'] = past_events.iloc[past_events.shape[0]-1,22] + prob
      return event


