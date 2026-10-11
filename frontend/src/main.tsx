import React from 'react';
import {createRoot} from 'react-dom/client';
import '@fontsource/fira-sans/latin-400.css';
import '@fontsource/fira-sans/latin-500.css';
import '@fontsource/fira-sans/latin-600.css';
import '@fontsource/fira-sans/latin-700.css';
import 'leaflet/dist/leaflet.css';
import './styles.css';
import App from './App';

class ErrorBoundary extends React.Component<{children:React.ReactNode},{error:boolean}>{
  state={error:false};
  static getDerivedStateFromError(){return {error:true};}
  render(){return this.state.error?<main className="crash"><h1>The dashboard needs a refresh</h1><p>Your experiment evidence is saved by the local service.</p><button onClick={()=>location.reload()}>Reload dashboard</button></main>:this.props.children;}
}
createRoot(document.getElementById('root')!).render(<ErrorBoundary><App/></ErrorBoundary>);
