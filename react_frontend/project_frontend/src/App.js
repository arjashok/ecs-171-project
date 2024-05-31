import React from "react";
import FeatureForm from './Components/FeatureForm';

function App() {
  return (
    <div>
      <div style={{backgroundColor:"blue"}}>
        <h1 style={{display: "flex", justifyContent: "center", color: "white"}}>Diabetes Classifier</h1>
        <h2 style={{display: "flex", justifyContent: "center", color: "white"}}>Group 6: Tej Sidhu, Arjun Ashok, Ayush Tripathi, Devon Streelman, Taha Abdullah</h2>
        <br />
        <p style={{display: "flex", justifyContent: "center", color: "white"}}>
          Classification Response will be at the bottom of the page under submit. <br />
          Click submit to get prediction
        </p>
      </div>
      <FeatureForm/>
    </div>
  );
}

export default App;
