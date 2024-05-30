import React, {useState} from 'react';
import Axios from 'axios';


/* const FeatureForm = ({onSubmit}) => {
    const [features, setFeatures] = useState({
        high_bp: '',
        high_chol: '',
        chol_check: '',
        bmi: '',
        smoker: '',
        stroke: '',
        heart_disease: '',
        physical_activity: '',
        fruits: '',
        veggies: '',
        heavy_drinker: '',
        no_doc_bc_cost: '',
        general_health: '',
        mental_health: '',
        physical_health: '',
        diff_walk: '',
        sex: '',
        age: '',
        education: '',
        income: ''
    });

    const handleChange = (e) => {
        const {name, value} = e.target;
        setFeatures({
            ...features,
            [name]: value,
        });
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        onSubmit(features);
    };

    return (
        <form onSubmit={handleSubmit}>
            <div>
                <label>high_bp</label>
                <input type="text" name="high_bp" value={features.high_bp} onChange={handleChange}/>
            </div>
            <div>
                <label>high_chol</label>
                <input type="text" name="high_chol" value={features.high_chol} onChange={handleChange}/>
            </div>
            <div>
                <label>chol_check</label>
                <input type="text" name="chol_check" value={features.chol_check} onChange={handleChange}/>
            </div>
            <div>
                <label>bmi</label>
                <input type="text" name="bmi" value={features.bmi} onChange={handleChange}/>
            </div>
            <div>
                <label>smoker</label>
                <input type="text" name="smoker" value={features.smoker} onChange={handleChange}/>
            </div>
            <div>
                <label>stroke</label>
                <input type="text" name="stroke" value={features.stroke} onChange={handleChange}/>
            </div>
            <div>
                <label>heart_disease</label>
                <input type="text" name="heart_disease" value={features.heart_disease} onChange={handleChange}/>
            </div>
            <div>
                <label>physical_activity</label>
                <input type="text" name="physical_activity" value={features.physical_activity} onChange={handleChange}/>
            </div>
            <div>
                <label>fruits</label>
                <input type="text" name="fruits" value={features.fruits} onChange={handleChange}/>
            </div>
            <div>
                <label>veggies</label>
                <input type="text" name="veggies" value={features.veggies} onChange={handleChange}/>
            </div>
            <div>
                <label>heavy_drinker</label>
                <input type="text" name="heavy_drinker" value={features.heavy_drinker} onChange={handleChange}/>
            </div>
            <div>
                <label>no_doc_bc_cost</label>
                <input type="text" name="no_doc_bc_cost" value={features.no_doc_bc_cost} onChange={handleChange}/>
            </div>
            <div>
                <label>general_health</label>
                <input type="text" name="general_health" value={features.general_health} onChange={handleChange}/>
            </div>
            <div>
                <label>mental_health</label>
                <input type="text" name="mental_health" value={features.mental_health} onChange={handleChange}/>
            </div>
            <div>
                <label>physical_health</label>
                <input type="text" name="physical_health" value={features.physical_health} onChange={handleChange}/>
            </div>
            <div>
                <label>diff_walk</label>
                <input type="text" name="diff_walk" value={features.diff_walk} onChange={handleChange}/>
            </div>
            <div>
                <label>sex</label>
                <input type="text" name="sex" value={features.sex} onChange={handleChange}/>
            </div>
            <div>
                <label>age</label>
                <input type="text" name="age" value={features.age} onChange={handleChange}/>
            </div>
            <div>
                <label>education</label>
                <input type="text" name="education" value={features.education} onChange={handleChange}/>
            </div>
            <div>
                <label>income</label>
                <input type="text" name="income" value={features.income} onChange={handleChange}/>
            </div>
        </form>
    );

    
};

export default FeatureForm;