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
            <form onSubmit={(e)=>submit(e)}>
                <div>
                    <label>high_bp:</label>
                    <input onChange={(e)=>handle(e)} id = "high_bp" value={data.high_bp} placeholder="high_bp" type="text" name="high_bp"></input>
                </div>
                <div>
                    <label>high_chol:</label>
                    <input onChange={(e)=>handle(e)} id = "high_chol" value={data.high_chol} placeholder = "high_chol" type="text" name="high_chol"></input>
                </div>
                <div>
                    <label>chol_check:</label>
                    <input onChange={(e)=>handle(e)} id = "chol_check" value={data.chol_check} placeholder = "chol_check" type="text" name="chol_check"></input>
                </div>
                <div>
                    <label>bmi:</label>
                    <input onChange={(e)=>handle(e)} id = "bmi" value={data.bmi} placeholder = "bmi" type="text" name="bmi"></input>
                </div>
                <div>
                    <label>smoker:</label>
                    <input onChange={(e)=>handle(e)} id = "smoker" value={data.smoker} placeholder = "smoker" type="text" name="smoker"></input>
                </div>
                <div>
                    <label>stroke:</label>
                    <input onChange={(e)=>handle(e)} id = "stroke" value={data.stroke} placeholder = "stroke" type="text" name="stroke"></input>
                </div>
                <div>
                    <label>heart_disease:</label>
                    <input onChange={(e)=>handle(e)} id = "heart_disease" value={data.heart_disease} placeholder = "heart_disease" type="text" name="heart_disease"></input>
                </div>
                <div>
                    <label>physical_activity:</label>
                    <input onChange={(e)=>handle(e)} id = "physical_activity" value={data.physical_activity} placeholder = "physical_activity" type="text" name="physical_activity"></input>
                </div>
                <div>
                    <label>fruits:</label>
                    <input onChange={(e)=>handle(e)} id = "fruits" value={data.fruits} placeholder = "fruits" type="text" name="fruits"></input>
                </div>
                <div>
                    <label>veggies:</label>
                    <input onChange={(e)=>handle(e)} id = "veggies" value={data.veggies} placeholder = "veggies" type="text" name="veggies"></input>
                </div>
                <div>
                    <label>heavy_drinker:</label>
                    <input onChange={(e)=>handle(e)} id = "heavy_drinker" value={data.heavy_drinker} placeholder = "heavy_drinker" type="text" name="heavy_drinker"></input>
                </div>
                <div>
                    <label>no_doc_bc_cost:</label>
                    <input onChange={(e)=>handle(e)} id = "no_doc_bc_cost" value={data.no_doc_bc_cost} placeholder = "no_doc_bc_cost" type="text" name="no_doc_bc_cost"></input>
                </div>
                <div>
                    <label>general_health:</label>
                    <input onChange={(e)=>handle(e)} id = "general_health" value={data.general_health} placeholder = "general_health" type="text" name="general_health"></input>
                </div>
                <div>
                    <label>mental_health:</label>
                    <input onChange={(e)=>handle(e)} id = "mental_health" value={data.mental_health} placeholder = "mental_health" type="text" name="mental_health"></input>
                </div>
                <div>
                    <label>physical_health:</label>
                    <input onChange={(e)=>handle(e)} id = "physical_health" value={data.physical_health} placeholder = "physical_health" type="text" name="physical_health"></input>
                </div>
                <div>
                    <label>diff_walk:</label>
                    <input onChange={(e)=>handle(e)} id = "diff_walk" value={data.diff_walk} placeholder = "diff_walk" type="text" name="diff_walk"></input>
                </div>
                <div>
                    <label>sex:</label>
                    <input onChange={(e)=>handle(e)} id = "sex" value={data.sex} placeholder = "sex" type="text" name="sex"></input>
                </div>
                <div>
                    <label>age:</label>
                    <input onChange={(e)=>handle(e)} id = "age" value={data.age} placeholder = "age" type="text" name="age"></input>
                </div>
                <div>
                    <label>education:</label>
                    <input onChange={(e)=>handle(e)} id = "education" value={data.education} placeholder = "education" type="text" name="education"></input>
                </div>
                <div>
                    <label>income:</label>
                    <input onChange={(e)=>handle(e)} id = "income" value={data.income} placeholder = "income" type="text" name="income"></input>
                </div>
                <div>
                    <button>Submit</button>
                </div>
            </form>
        </div>
    );
}

export default FeatureForm;