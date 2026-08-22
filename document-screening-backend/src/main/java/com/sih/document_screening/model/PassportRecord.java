package com.sih.document_screening.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import java.time.LocalDate;

@Entity
@Table(name = "passport_records")
@Getter
@Setter
@NoArgsConstructor
public class PassportRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @OneToOne
    @JoinColumn(name = "screening_id", nullable = false)
    private Screening screening;

    private String documentNumber;
    private String fullName;
    private LocalDate dateOfBirth;
    private LocalDate expiryDate;
    private String nationality;

    @Column(columnDefinition = "TEXT")
    private String rawMrz;
    
    private Boolean mrzChecksumValid;
    private Double ocrConfidence;
}