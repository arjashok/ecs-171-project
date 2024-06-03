import React, {useState} from "react";
import Axios from "axios";

function FeatureForm(){
    const url = "http://127.0.0.1:5000/model/predict"
    const [data, setData] = useState({
        high_bp: "",
        high_chol: "",
        chol_check: "",
        bmi: "",
        smoker: "",
        stroke: "",
        heart_disease: "",
        physical_activity: "",
        fruits: "",
        veggies: "",
        heavy_drinker: "",
        healthcare: "",
        no_doc_bc_cost: "",
        general_health: "",
        mental_health: "",
        physical_health: "",
        diff_walk: "",
        sex: "",
        age: "",
        education: "",
        income: ""
    });

    const [response, setResponse] = useState(null);

    function submit(e){
        e.preventDefault();
        console.log()
        console.log("Data sent:", data);
        Axios.post(url, {
            high_bp: data.high_bp,
            high_chol: data.high_chol,
            chol_check: data.chol_check,
            bmi: data.bmi,
            smoker: data.smoker,
            stroke: data.stroke,
            heart_disease: data.heart_disease,
            physical_activity: data.physical_activity,
            fruits: data.fruits,
            veggies: data.veggies,
            heavy_drinker: data.heavy_drinker,
            healthcare: data.healthcare,
            no_doc_bc_cost: data.no_doc_bc_cost,
            general_health: data.general_health,
            mental_health: data.mental_health,
            physical_health: data.physical_health,
            diff_walk: data.diff_walk,
            sex: data.sex,
            age: data.age,
            education: data.education,
            income: data.income
        }).then(
            res=> {
                console.log(res.data);
                setResponse(res.data);
            }
        ).catch(
            error => {
                console.error("Error: ", error);
            });
    }

    function handle(e){
        const newdata={...data}
        newdata[e.target.id] = e.target.value
        setData(newdata)
        console.log(newdata)
    }

    return (
        <div>
        <div style={{display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
            <form onSubmit={(e)=>submit(e)}>
                <div style={{marginBottom: "15px"}}>
                    <label>Do you have high blood pressure? (0: no | 1: yes):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "high_bp" value={data.high_bp} placeholder="high_bp" type="text" name="high_bp"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>Do you have high cholestral? (0: no | 1: yes):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "high_chol" value={data.high_chol} placeholder = "high_chol" type="text" name="high_chol"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>Have you had a cholestral check in the past 5 years? (0: no | 1: yes):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "chol_check" value={data.chol_check} placeholder = "chol_check" type="text" name="chol_check"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>What is your BMI? (0-100):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "bmi" value={data.bmi} placeholder = "bmi" type="text" name="bmi"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>Have you smoked 100 cigarettes in your life? (0: no | 1: yes):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "smoker" value={data.smoker} placeholder = "smoker" type="text" name="smoker"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>Have you had a stroke? (0: no | 1: yes):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "stroke" value={data.stroke} placeholder = "stroke" type="text" name="stroke"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>Do you have a heart disease? (0: no | 1: yes):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "heart_disease" value={data.heart_disease} placeholder = "heart_disease" type="text" name="heart_disease"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>Have you done physical activity in past 30 days? (0: no | 1: yes):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "physical_activity" value={data.physical_activity} placeholder = "physical_activity" type="text" name="physical_activity"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>Do you consume fruits daily? (0: no | 1: yes):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "fruits" value={data.fruits} placeholder = "fruits" type="text" name="fruits"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>Do you consume veggies daily? (0: no | 1: yes):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "veggies" value={data.veggies} placeholder = "veggies" type="text" name="veggies"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>Are you a heavy drinker? (Men: More than 14 drinks Weekly, Women: More than 7 drinks Weekly) (0: no | 1: yes):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "heavy_drinker" value={data.heavy_drinker} placeholder = "heavy_drinker" type="text" name="heavy_drinker"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>Do you have any kind of healthcare coverage? (0: no | 1: yes):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "healthcare" value={data.healthcare} placeholder = "healthcare" type="text" name="healthcare"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>In the past 12 months, have you wanted to see a doctor but couldn't because of the cost? (0: no | 1: yes):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "no_doc_bc_cost" value={data.no_doc_bc_cost} placeholder = "no_doc_bc_cost" type="text" name="no_doc_bc_cost"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>In general, how good is your health? (excellent | very good | good | fair | poor):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "general_health" value={data.general_health} placeholder = "general_health" type="text" name="general_health"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>How many days in the past month was your mental health not good? (0-30):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "mental_health" value={data.mental_health} placeholder = "mental_health" type="text" name="mental_health"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>How many days in the past month was your physical health not good? (0-30):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "physical_health" value={data.physical_health} placeholder = "physical_health" type="text" name="physical_health"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>Do you have serious difficulty walking or climbing stairs? (0: no | 1: yes):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "diff_walk" value={data.diff_walk} placeholder = "diff_walk" type="text" name="diff_walk"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>What is your biological sex? (0: female | 1: male)</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "sex" value={data.sex} placeholder = "sex" type="text" name="sex"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>What is your age? (18-99):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "age" value={data.age} placeholder = "age" type="text" name="age"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>What is your education level? (No Education | Elementary School | Some High School | High School Graduate | Some College or Technical School | College Graduate):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "education" value={data.education} placeholder = "education" type="text" name="education"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <label>What is your annual income? (Input a Number without a $):</label>
                    <br />
                    <input onChange={(e)=>handle(e)} id = "income" value={data.income} placeholder = "income" type="text" name="income"></input>
                </div>
                <div style={{marginBottom: "15px"}}>
                    <button style={{ backgroundColor: "blue", color: "white", padding: "10px 20px", borderRadius: "5px" }}>Submit</button>
                </div>
            </form>
        </div>

            <div  style={{ marginTop: "20px", textAlign: "left", paddingLeft: "20px" }}>
                {response && (
                    <div>
                        {response.error && response.message ? (
                            <div>
                                <h3 style={{ color: "red" }}>{response.message}</h3>
                                <p style={{ color: "red" }}>{response.error}</p>
                            </div>
                        ): (
                            response.prediction && response.confidence && response.analysis && (
                                <div>
                                    <h3>Prediction Details:</h3>
                                    <p>Prediction: {response.prediction}</p>
                                    <p>Confidence: {response.confidence}</p>
                                    <p>Analysis: {response.analysis}</p>
                                </div>
                            )
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

export default FeatureForm;