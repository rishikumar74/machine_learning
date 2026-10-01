<%@ Page Title="Home" Language="C#" MasterPageFile="~/Site1.Master" AutoEventWireup="true" CodeBehind="Home.aspx.cs" Inherits="WebApplication11.Home" %>

<asp:Content ID="Content1" ContentPlaceHolderID="head" runat="server">
</asp:Content>

<asp:Content ID="Content2" ContentPlaceHolderID="ContentPlaceHolder1" runat="server">
    <!-- Picture Banner -->
    <div class="cvr-banner">
        <asp:Image ID="imgCampus" runat="server" 
                   ImageUrl="https://images.unsplash.com/photo-1562774053-701939374585?auto=format&fit=crop&w=1000&q=80" 
                   AlternateText="CVR Campus" 
                   CssClass="banner-img" />
        <div class="banner-overlay">
            <h2>Welcome to CVR Student Management Portal</h2>
            <p>Vastunagar, Mangalpalli (V), Ibrahimpatnam, Hyderabad</p>
        </div>
    </div>

    <!-- Society Overview Header Text -->
    <div class="society-info">
        <h3>About Cherabuddi Education Society</h3>
        <p>
            The Cherabuddi Education Society was registered in January 1999 as an NRI-promoted society. 
            The Society aims at creating a state-of-the-art engineering institution in association with 
            leading NRI technology professionals and well-known academicians of the twin states of 
            Telangana &amp; Andhra Pradesh. The Society aims to harness technical excellence with a 
            commitment to the ethos of 'useful learning'.
        </p>
        <p>
            Our students remain at the heart of our vision, benefiting from our innovative teaching methods, 
            and close links with premier companies and research institutions in India and abroad. 
            Students will enjoy engineering and will have the chance to develop skills that will last a lifetime.
        </p>
    </div>

    <!-- Metrics -->
    <div class="dashboard-grid">
        <div class="dash-card">
            <h4>B.Tech Branches</h4>
            <div class="metric-number">9</div>
            <span>CSE, AI&amp;ML, DS, IT, ECE, etc.</span>
        </div>
        <div class="dash-card">
            <h4>Placement Offers</h4>
            <div class="metric-number">1,250+</div>
            <span>Current Academic Year</span>
        </div>
        <div class="dash-card">
            <h4>Student Enrollment</h4>
            <div class="metric-number">4,200+</div>
            <span>UG &amp; PG Students</span>
        </div>
    </div>

    <!-- Quick Navigation HyperLinks -->
    <div>
        <h3 style="color:#2a9df4; margin-bottom:10px;">Quick Navigation</h3>
        <div class="action-buttons">
            <asp:HyperLink ID="lnkReg" runat="server" NavigateUrl="~/StudentRegistration.aspx" CssClass="portal-btn">
                Enroll Student
            </asp:HyperLink>
            <asp:HyperLink ID="lnkDir" runat="server" NavigateUrl="~/StudentList.aspx" CssClass="portal-btn">
                Student Directory
            </asp:HyperLink>
            <asp:HyperLink ID="lnkCourse" runat="server" NavigateUrl="~/Courses.aspx" CssClass="portal-btn">
                Courses Offered
            </asp:HyperLink>
            <asp:HyperLink ID="lnkCont" runat="server" NavigateUrl="~/ContactUs.aspx" CssClass="portal-btn">
                Contact Campus
            </asp:HyperLink>
        </div>
    </div>
</asp:Content>