import React, { useState } from "react";
import FileUpload from "@/components/FileUpload";
import ProcessButton from "@/components/ProcessButton";

function home() {
  const [resPath, setResPath] = useState<string>("");

  return (
    <div className="container">
      <h1 className="header">Home</h1>
      <div style={{ flexDirection: "row" }}>
        <FileUpload type="cold" />
        <FileUpload type="hot" />
        <ProcessButton setResultPath={setResPath} />
      </div>
      {resPath && <img src={resPath} alt="Result Image" className="image" />}
    </div>
  );
}

export default home;
