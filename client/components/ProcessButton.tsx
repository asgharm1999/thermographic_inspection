import React from 'react';

// interface ProcessButtonProps {
//     setResultPath: (resultPath: string) => void;
// }

const ProcessButton = ({ setResultPath }) => {
    const fetchData = async () => {
        try {
            const response = await fetch('http://localhost:8080/preprocess');
            const data = await response.json();

            setResultPath(data.resultPath);
        } catch (error) {
            console.error('Error:', error);
        }
    };

    return <button onClick={fetchData}>Process Data</button>;
};

export default ProcessButton;
