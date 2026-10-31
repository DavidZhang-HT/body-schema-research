"""One learned imagination step, using a genuine held-out exploration input."""
from ground_model import *
if __name__=='__main__':
 a=np.load(OUT/'exploration.npz');i=int(np.where(a['case']>=20)[0][0]);x=a['x'][i:i+1];u=a['u'][i:i+1]
 predicted,uncertainty=Ensemble().predict(x,u);actual=a['y'][i];nodes=predicted[0,:375].reshape(25,15)
 print(json.dumps({'horizon_seconds':DURATION,'schema_action':u[0].tolist(),'com_height_now_m':float(x[0,394]),'com_height_predicted_m':float(predicted[0,394]),'com_height_observed_m':float(actual[394]),'ensemble_com_height_std_m':float(uncertainty[0,394]),'predicted_left_right_foot_vertical_force_N':nodes[[6,12],9].tolist(),'predicted_ground_joint_loads_Nm':nodes[1:,3].tolist(),'note':'Ideal simulation inputs. Ensemble disagreement is not a calibrated confidence interval.'},indent=2))
