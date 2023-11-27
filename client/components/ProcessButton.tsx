import React from "react";
import styles from "./ProcessButton.module.css";

const methodAndOptions = [
  {
    method: "PCT",
    options: ["numEOFs"],
  },
  {
    method: "SPCT",
    options: ["numEOFs"],
  },
];

const ProcessButton = ({
  setResultPath,
}: {
  setResultPath: React.Dispatch<React.SetStateAction<string>>;
}) => {
  const [method, setMethod] = React.useState<string>("PCT");
  const [options, setOptions] = React.useState<{ [key: string]: string }>({});

  const fetchData = async () => {
    try {
      const formData = new FormData();
      formData.append("method", method);
      formData.append("options", JSON.stringify(options));

      const response = await fetch("http://localhost:8080/preprocess", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      console.log("Process result: ", data);
      setResultPath(data.resultPath);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  return (
    <div className={styles.container}>
      <select
        onChange={(e) => setMethod(e.target.value)}
        className={styles.select}
      >
        <option value="PCT">PCT</option>
        <option value="SPCT">SPCT</option>
      </select>

      {methodAndOptions.map((item) => {
        if (item.method === method) {
          return (
            <div className={styles.optionsContainer}>
              {item.options.map((option) => (
                <div>
                  <label>{option}: </label>
                  <input
                    type="text"
                    className={styles.input}
                    onChange={(e) =>
                      setOptions({ ...options, [option]: e.target.value })
                    }
                  />
                </div>
              ))}
            </div>
          );
        }
      })}

      <button onClick={fetchData} className={styles.button}>
        Process Data
      </button>
    </div>
  );
};

export default ProcessButton;
