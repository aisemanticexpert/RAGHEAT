import React from 'react';
import HeatMapDashboard from './components/HeatMapDashboard';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>RAGHeat - Real-time Stock Recommendation System</h1>
      </header>
      <main>
        <HeatMapDashboard />
      </main>
    </div>
  );
}

export default App;