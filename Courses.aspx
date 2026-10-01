<%@ Page Title="Courses & Branches" Language="C#" MasterPageFile="~/Site1.Master" AutoEventWireup="true" CodeBehind="Courses.aspx.cs" Inherits="WebApplication11.Courses" %>

<asp:Content ID="Content1" ContentPlaceHolderID="head" runat="server">
</asp:Content>

<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder1" runat="server">
    <h2>Academic Departments &amp; Courses</h2>
    <p>CVR College of Engineering provides specialized autonomous curriculum designed with industry inputs:</p>

    <div style="margin-top:20px; display:flex; flex-direction:column; gap:15px;">
        <div class="dash-card" style="text-align:left;">
            <h4 style="color:#2a9df4; font-size:15px;">Department of Computer Science &amp; Engineering (CSE)</h4>
            <p style="font-size:13px; color:#555; margin-top:5px;">Focus on Cloud Computing, Full-Stack, and Operating Systems.</p>
        </div>
        <div class="dash-card" style="text-align:left;">
            <h4 style="color:#2a9df4; font-size:15px;">Department of AI &amp; Machine Learning (CSM)</h4>
            <p style="font-size:13px; color:#555; margin-top:5px;">Deep Learning, Computer Vision, and Neural Networks.</p>
        </div>
        <div class="dash-card" style="text-align:left;">
            <h4 style="color:#2a9df4; font-size:15px;">Department of Electronics &amp; Communication (ECE)</h4>
            <p style="font-size:13px; color:#555; margin-top:5px;">VLSI Design, Embedded Systems, and IoT Applications.</p>
        </div>
    </div>
</asp:Content>