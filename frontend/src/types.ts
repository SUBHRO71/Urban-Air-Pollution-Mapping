export type Dataset = {id:string;name:string;source:string;description:string;status:string;action?:string;source_url:string;rows?:number;time_steps?:number;start?:string;end?:string;checksum?:string;features?:string[];targets?:Record<string,string>;stations?:string[];coordinates?:{station:string;latitude:number;longitude:number}[];graph_ready?:boolean;grid_shape?:number[];missing_percent?:Record<string,number>;frequency?:string;license?:string};
export type Model = {id:string;name:string;group:string;architecture:string;compatible:boolean;reason:string|null;modality:string;implementation:string;aliases:string[];runtime:string};
export type Metrics = {mae:number|null;rmse:number|null;r2:number|null;mse?:number|null;mape?:number|null;smape?:number|null;skill?:number|null;n?:number;coverage?:number;[key:string]:unknown};
export type Run = {id:string;kind:string;status:string;parent:string|null;created:string;updated:string;payload:Config;result:any;events?:{time:string;message:string}[]};
export type Config = {dataset:string;model:string;target:string;evaluation:string;lookback:number;horizon:number;epochs:number;patience:number;hidden:number;batch_size:number;seed:number;preset:string;device:string;max_origins:number;train_ratio:number;learning_rate:number;stations:string[];feature_columns:string[]};
export type Point = {timestamp:string;observed:number|null;predicted?:number|null;station?:string;fold?:string};
export type Preview = {points:Point[];stations:{station:string;latitude?:number;longitude?:number;mean:number|null;count:number}[];target:string;unit:string;summary:string};

export async function api<T=any>(url:string,options:RequestInit={}):Promise<T>{
  const response=await fetch('/api'+url,options);
  if(!response.ok){const error=await response.json().catch(()=>({detail:response.statusText}));throw new Error(typeof error.detail==='string'?error.detail:JSON.stringify(error.detail));}
  return response.json();
}
export const post=<T=any>(url:string,data:unknown)=>api<T>(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
export function number(value:unknown,digits=3):string{return typeof value==='number'&&Number.isFinite(value)?value.toLocaleString('en-IN',{maximumFractionDigits:digits}):'N/A';}
export function date(value?:string,full=false):string{return value?new Date(value).toLocaleString('en-IN',{timeZone:'UTC',day:'2-digit',month:'short',year:full?'numeric':undefined,...(full?{hour:'2-digit',minute:'2-digit'}:{})}):'—';}
export const defaultConfig:Config={dataset:'',model:'random_forest',target:'CO',evaluation:'both',lookback:24,horizon:1,epochs:8,patience:3,hidden:32,batch_size:64,seed:42,preset:'quick',device:'cpu',max_origins:1200,train_ratio:.8,learning_rate:.001,stations:[],feature_columns:[]};
