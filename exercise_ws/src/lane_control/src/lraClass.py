# ---------------------------------------------------------------------------------
#
#
#
#                           DO NOT EDIT THE LINES BELOW
#
#
#
# ---------------------------------------------------------------------------------


class LRA2_HELPER():
    
    def __init__(self):
        self.d=[]
        self.phi=[]
        self.u=[]

    def wrapEnv(self,env):
        from gym import wrappers
        return wrappers.Monitor(env, "./gym-results", force=True)

    def renderEnv(self, env):
        import io
        import base64
        from IPython.display import HTML

        video = io.open('./gym-results/openaigym.video.%s.video000000.mp4' % env.file_infix, 'r+b').read()
        encoded = base64.b64encode(video)
        return HTML(data='''
            <video width="360" height="auto" alt="test" controls><source src="data:video/mp4;base64,{0}" type="video/mp4" /></video>'''
        .format(encoded.decode('ascii')))


    def solveRiccati(self, A, B, Q, R):
        """Solving the Riccati equation
            - A, B: system matrices for x and u respectivelly
            - Q, R: cost matrices for x and u respectivelly
        """
        from scipy import linalg
        return linalg.solve_discrete_are(A,B,Q,R)

    def loadData(self, file_name):
        """Save the control and the state data:
            - file_name: csv file with state and control data
            - format: x, u
        """
        import pandas as pd

        data = pd.read_csv(file_name) 
        return data

    def collectData(self,u,x):
        """collect the control input and the state data:
            - u is a real valued scalar
            - x is an array of two real numbers
        """
        self.d.append(x[0])
        self.phi.append(x[1])
        self.u.append(u)

    def saveData(self):
        """Save the control and the state data:
            - csv file: model_data.csv
            - format: x, u
        """
        import pandas as pd
        # Set up empty DataFrame
        data = pd.DataFrame()

        data['d'] = self.d
        data['phi'] = self.phi
        data['u'] = self.u

        data.to_csv('model_data.csv', index=False, header=True)
