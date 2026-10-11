
from copy import deepcopy
import math
import numpy as np
import params as settings

DEVICE="cuda:0"
PREDICT_CHECKPOINT=None
PREDICT_OUTPUT="runs/geometry_unseen_corrected"
PREDICT_SIDES=[13,16,20,32]
PREDICT_AMPLITUDE=.0015
PREDICT_TIMES=[10.,30.,50.,100.]
PREDICT_LOG_EVERY=20
PLOT_DISCARD_TIME=50.
VORTICITY_START=1.
VORTICITY_END=100.
VORTICITY_INTERVAL=1.
VORTICITY_FPS=10
MAKE_VORTICITY_GIF=True
REPLOT_DIRECTORY=None
CORRECTION_MAX_ITERATIONS=1500
CORRECTION_LR=.0001
CORRECTION_CHECK_EVERY=25


def correct_step(physics,model,optimizer,previous,reynolds,time,config,out,step):
    import torch
    from model import append_csv
    previous=previous.detach(); model.eval()
    with torch.no_grad():
        velocity,pressure=physics.predict(model,previous,reynolds,time)
        _,before=physics.residual(previous,velocity,pressure,reynolds)
    if physics.accepted(before): return velocity,pressure,before,before,0,True,"direct_pass"
    print("Correct geometry",physics.sides,"step",step,"initial residual:",before,flush=True)
    append_csv(out/"correction_iterations.csv",{"step":step,"time":time+config["physics"]["dt"],
               "updates":0,"passed":False,"reason":"initial_gate_failed",**before})
    updates,reason=0,"correction_budget_exhausted"; model.train()
    with torch.enable_grad():
        for iteration in range(1,CORRECTION_MAX_ITERATIONS+1):
            optimizer.zero_grad(set_to_none=True)
            candidate,p_candidate=physics.predict(model,previous,reynolds,time)
            loss,_=physics.residual(previous,candidate,p_candidate,reynolds)
            if not torch.isfinite(loss): reason="nonfinite_loss"; break
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config["training"]["clip_grad"])
            if not torch.isfinite(norm): reason="nonfinite_gradient"; break
            optimizer.step(); updates+=1
            if iteration%CORRECTION_CHECK_EVERY==0 or iteration==CORRECTION_MAX_ITERATIONS:
                model.eval()
                with torch.no_grad():
                    velocity,pressure=physics.predict(model,previous,reynolds,time)
                    _,metrics=physics.residual(previous,velocity,pressure,reynolds)
                passed=physics.accepted(metrics)
                append_csv(out/"correction_iterations.csv",{"step":step,"time":time+config["physics"]["dt"],
                           "updates":updates,"passed":passed,"reason":"corrected_pass" if passed else "optimizing",**metrics})
                print("Correction",step,"updates",updates,metrics,flush=True)
                if passed:
                    optimizer.zero_grad(set_to_none=True)
                    return velocity.detach(),pressure.detach(),metrics,before,updates,True,"corrected_pass"
                model.train()
    model.eval(); optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        velocity,pressure=physics.predict(model,previous,reynolds,time)
        _,metrics=physics.residual(previous,velocity,pressure,reynolds)
    append_csv(out/"correction_iterations.csv",{"step":step,"time":time+config["physics"]["dt"],
               "updates":updates,"passed":False,"reason":reason,**metrics})
    return velocity,pressure,metrics,before,updates,False,reason


def plot_suite(root):
    import predict
    # Pass this entry's own settings explicitly; predict.py's user constants stay independent.
    options={"sides":None,"times":PREDICT_TIMES,"discard_time":PLOT_DISCARD_TIME,
             "sequence_start":VORTICITY_START,"sequence_end":VORTICITY_END,
             "sequence_interval":VORTICITY_INTERVAL,"fps":VORTICITY_FPS,"gif":MAKE_VORTICITY_GIF}
    root=settings.resolve_path(root)
    cases=[root] if(root/"history.csv").exists() else sorted(p for p in root.iterdir() if p.is_dir() and(p/"history.csv").exists())
    if not cases: print("No accepted results are available for plotting:",root); return False
    for case in cases: predict.plot_training_results(case,options=options)
    # A single across-geometry probe comparison and combined force workbook.
    predict.export_comparisons(cases,root/"geometry_comparison",discard_time=PLOT_DISCARD_TIME)
    return True


def rollout():
    import torch
    from model import FNO,CylinderPhysics,get_device,validate,append_csv,write_json,record,load_checkpoint,shape_name
    from predict import requested_times
    checkpoint_path=settings.resolve_path(PREDICT_CHECKPOINT) if PREDICT_CHECKPOINT else settings.latest_training_directory()/"checkpoint.pt"
    saved=load_checkpoint(checkpoint_path); training=saved["config"]; trained_sides=training["geometry"]["sides"]
    if not PREDICT_SIDES or len(set(PREDICT_SIDES))!=len(PREDICT_SIDES) or any(not isinstance(n,int) or isinstance(n,bool) or n!=0 and n<3 for n in PREDICT_SIDES):
        raise ValueError("Unique integer sides>=3 or0 required")
    if set(PREDICT_SIDES)&set(trained_sides): raise ValueError("This file is intended only for unseen geometries. Remove the trained side counts.")
    if not math.isfinite(PREDICT_AMPLITUDE) or not PREDICT_TIMES or PREDICT_LOG_EVERY<1:
        raise ValueError("Invalid prediction settings")
    if CORRECTION_MAX_ITERATIONS<1 or CORRECTION_CHECK_EVERY<1 or not math.isfinite(CORRECTION_LR) or CORRECTION_LR<=0:
        raise ValueError("Invalid correction budget")
    dt=training["physics"]["dt"]
    sequence=requested_times(VORTICITY_START,VORTICITY_END,VORTICITY_INTERVAL)
    targets=sorted(set(sequence+list(PREDICT_TIMES))); target_steps=set()
    for t in targets:
        if not math.isfinite(t) or t<=0 or not math.isclose(round(t/dt)*dt,t,abs_tol=1e-7,rel_tol=0):
            raise ValueError("All requested times must be positive multiples of checkpoint dt")
        target_steps.add(round(t/dt))
    max_step=max(target_steps); device=get_device(DEVICE)
    root=settings.fresh_output_directory(PREDICT_OUTPUT)
    write_json(root/"suite.json",{"checkpoint":str(checkpoint_path),"training_config":training,
               "sides":PREDICT_SIDES,"Re":100,"times":targets,"mode":"physics_corrected_new_geometry_solve",
               "note":"Per-step optimization on unseen geometry, not frozen operator generalization"})
    all_passed=True
    with torch.no_grad():
        for n in PREDICT_SIDES:
            c=deepcopy(training); c["geometry"]["sides"]=[n]; c["cases"]["amplitude"]=[PREDICT_AMPLITUDE]; validate(c)
            out=root/shape_name(n); out.mkdir(); write_json(out/"config_used.json",c)
            physics=CylinderPhysics(c,device); re=physics.reynolds()
            model=FNO(**c["model"]).to(device); model.load_state_dict(saved["model"])
            optimizer=torch.optim.Adam(model.parameters(),lr=CORRECTION_LR)
            state=physics.initial(torch.tensor([PREDICT_AMPLITUDE],dtype=torch.float32,device=device))
            accepted,total_updates,corrected,direct=0,0,0,0; reason="completed"; metrics={}
            for step in range(max_step):
                velocity,pressure,metrics,before,updates,passed,reason=correct_step(physics,model,optimizer,state,re,step*dt,c,out,step+1)
                total_updates+=updates; corrected+=int(updates>0); direct+=int(passed and updates==0)
                append_csv(out/"residuals.csv",{"step":step+1,"time":(step+1)*dt,"passed":passed,
                           "correction_updates":updates,"reason":reason,**{"before_"+k:v for k,v in before.items()},**metrics})
                if not passed:
                    all_passed=False; print("STOP",shape_name(n),step+1,metrics,flush=True); break
                previous=state; state=velocity.detach(); accepted=step+1
                record(out,physics,state,pressure,re,accepted*dt,accepted,accepted in target_steps,previous=previous)
                if accepted%PREDICT_LOG_EVERY==0: print(shape_name(n),"t*=",accepted*dt,flush=True)
            status={"sides":n,"shape":shape_name(n),"Re":100,"accepted_steps":accepted,
                    "total_correction_updates":total_updates,"corrected_steps":corrected,"direct_steps":direct,
                    "status":"completed" if accepted==max_step else reason,"last_metrics":metrics,
                    "missing_times":[t for t in targets if round(t/dt)>accepted]}
            write_json(out/"status.json",status)
            append_csv(root/"summary.csv",{k:v for k,v in status.items() if k not in("last_metrics","missing_times")})
    plot_suite(root)
    return (0 if all_passed else 2),root


if __name__=="__main__":
    if REPLOT_DIRECTORY is not None:
        raise SystemExit(0 if plot_suite(REPLOT_DIRECTORY) else 2)
    code,directory=rollout(); print("Unseen-geometry results directory:",directory,flush=True); raise SystemExit(code)
