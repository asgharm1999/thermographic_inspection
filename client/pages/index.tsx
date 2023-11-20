import React from "react";
import FileUpload from "@/components/FileUpload";

function home() {
  return (
    <div>
      <h1>Home</h1>
      <div style={{ flexDirection: "row" }}>
        <FileUpload type="cold" />
        <FileUpload type="hot" />
      </div>
    </div>
  );
}

export default home;
